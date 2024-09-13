"""
This script processes audio files to reduce noise and extract information.
It performs the following main tasks:
1. Applies basic noise reduction to the audio
2. Transcribes the denoised audio using AWS Transcribe
3. Extracts information from the transcription using AWS Comprehend and Bedrock
4. Calculates and reports audio quality metrics before and after processing
5. Handles AWS S3 operations for file storage and retrieval
"""

import numpy as np
import librosa
import soundfile as sf
import argparse
import json
import os
import logging
import boto3
from botocore.exceptions import ClientError
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
import time
import requests
import noisereduce as nr

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_file='config.json'):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)

def load_audio(file_path, sr=16000):
    """Load an audio file and resample if necessary."""
    logger.info(f"Loading audio file: {file_path}")
    audio, file_sr = librosa.load(file_path, sr=None)
    if file_sr != sr:
        logger.info(f"Resampling audio from {file_sr} Hz to {sr} Hz")
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio, sr

def save_audio(file_path, audio, sr):
    """Save audio to a file."""
    logger.info(f"Saving processed audio to: {file_path}")
    sf.write(file_path, audio, sr)

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio (SNR) in dB."""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

def denoise_audio(audio, sr):
    """Apply basic noise reduction to the audio."""
    logger.info("Applying noise reduction")
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    return reduced_noise

def calculate_audio_metrics(original, processed):
    """Calculate audio quality metrics."""
    noise = original - processed
    snr_original = calculate_snr(original, noise)
    snr_processed = calculate_snr(processed, noise)
    
    return {
        'original_snr': snr_original,
        'denoised_snr': snr_processed
    }

def create_s3_bucket_if_not_exists(bucket_name):
    """Create an S3 bucket if it doesn't already exist."""
    s3 = boto3.client('s3')
    try:
        s3.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} already exists")
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            logger.info(f"Creating bucket {bucket_name}")
            s3.create_bucket(Bucket=bucket_name)
        else:
            logger.error(f"Error checking bucket {bucket_name}: {e}")
            raise
    return bucket_name

def upload_file_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket."""
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        logger.info(f"File {file_name} uploaded to {bucket}/{object_name}")
    except ClientError as e:
        logger.error(f"Error uploading file to S3: {e}")
        return False
    return True

def transcribe_audio(audio_file, bucket_name):
    """Transcribe an audio file using AWS Transcribe."""
    logger.info(f"Starting transcription of {audio_file}")
    transcribe = boto3.client('transcribe')
    
    job_name = f"TranscribeJob_{int(time.time())}"
    object_name = os.path.basename(audio_file)
    job_uri = f"s3://{bucket_name}/{object_name}"
    
    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': job_uri},
            MediaFormat='wav',
            LanguageCode='hi-IN'  # Set to Hindi, change if needed
        )

        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)
    
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            result = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            transcript_file_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
            
            transcript_response = requests.get(transcript_file_uri)
            transcript = transcript_response.json()
            
            text = transcript['results']['transcripts'][0]['transcript']
            logger.info("Transcription successful")
            logger.info(f"Transcription: {text}")
            return text
        else:
            logger.error("Transcription failed")
            return ""
    
    except ClientError as e:
        logger.error(f"Error during transcription: {e}")
        return ""

def extract_information(text, config):
    """Extract information from transcribed text using AWS Comprehend and Bedrock."""
    if not text.strip():
        logger.warning("No transcription available. The audio might be silent or the speech recognition failed.")
        return "No transcription available. The audio might be silent or the speech recognition failed."

    logger.info("Extracting information from transcription")
    
    comprehend = boto3.client('comprehend')
    
    try:
        language_response = comprehend.detect_dominant_language(Text=text)
        dominant_language = language_response['Languages'][0]['LanguageCode']
        
        entities_response = comprehend.detect_entities(Text=text, LanguageCode=dominant_language)
        entities = [entity['Text'] for entity in entities_response['Entities'] if entity['Type'] in ['PERSON', 'ORGANIZATION', 'LOCATION']]
        
        logger.info(f"Detected language: {dominant_language}")
        logger.info(f"Detected entities: {entities}")
        
    except ClientError as e:
        logger.error(f"Error during entity extraction: {e}")
        entities = []
        dominant_language = "unknown"
    
    bedrock_chat = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0.5, "top_p": 0.9, "max_tokens": 500}
    )
    
    prompt_template = PromptTemplate(
        input_variables=["text", "entities", "language"],
        template=config['bedrock_prompt']
    )
    
    prompt = prompt_template.format(text=text, entities=", ".join(entities), language=dominant_language)

    try:
        response = bedrock_chat.invoke(prompt)
        logger.info("Information extraction successful")
        return response.content
    except Exception as error:
        logger.error(f"Error invoking Bedrock model: {error}")
        return f"Error in information extraction: {str(error)}"

def process_audio(input_file, output_file, bucket_name, config):
    """Process an audio file: denoise, transcribe, and extract information."""
    audio, sr = load_audio(input_file)
    logger.info(f"Audio length: {len(audio)/sr:.2f} seconds")
    
    denoised = denoise_audio(audio, sr)
    
    metrics = calculate_audio_metrics(audio, denoised)
    
    save_audio(output_file, denoised, sr)
    
    logger.info("Audio Quality Metrics:")
    logger.info(f"Original SNR: {metrics['original_snr']:.2f} dB")
    logger.info(f"Denoised SNR: {metrics['denoised_snr']:.2f} dB")
    
    snr_improvement = metrics['denoised_snr'] - metrics['original_snr']
    logger.info(f"SNR Improvement: {snr_improvement:.2f} dB ({(snr_improvement/metrics['original_snr']*100):.2f}%)")
    
    create_s3_bucket_if_not_exists(bucket_name)
    if not upload_file_to_s3(output_file, bucket_name):
        logger.error("Failed to upload file to S3. Exiting.")
        return
    
    transcription = transcribe_audio(output_file, bucket_name)
    logger.info(f"Transcription: {transcription}")
    
    information = extract_information(transcription, config)
    logger.info("Extracted Information:")
    logger.info(information)

def main():
    """Main function to process command-line arguments and run the audio processing pipeline."""
    parser = argparse.ArgumentParser(description="Process audio file to denoise, transcribe, and extract information using AWS services.")
    parser.add_argument("input_file", help="Path to the input audio file")
    parser.add_argument("output_file", help="Path to save the processed audio file")
    parser.add_argument("bucket_name", help="Name of the S3 bucket to use")
    parser.add_argument("--config", default="config.json", help="Path to the configuration file")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    logger.info(f"Processing file: {args.input_file}")
    logger.info(f"Saving processed audio to: {args.output_file}")
    logger.info(f"Using S3 bucket: {args.bucket_name}")
    
    process_audio(args.input_file, args.output_file, args.bucket_name, config)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()