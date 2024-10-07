# To call watsonx's LLM, we need to import the library of IBM Watson Machine Learning
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model

# placeholder for Watsonx_API and Project_id incase you need to use the code outside this environment
# API_KEY = "Your WatsonX API"
PROJECT_ID= "bb288bc7-5120-402b-9702-1a40948f1fa6"

# Define the credentials 
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "9pSyzJs7QSNN2UsPVj9L_q6wyD0znBvKv2PaVGae_QmB"
}
    
# Specify model_id that will be used for inferencing
model_id = ModelTypes.FLAN_UL2

# Define the model parameters
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

# Define the LLM
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)

import base64
import requests
import io
from pydub import AudioSegment
import subprocess


def convert_audio_format(audio_binary):
    try:
        input_file = "input_audio.wav"
        output_file = "output_audio.wav"

        # Save the audio to a file
        with open(input_file, "wb") as f:
            f.write(audio_binary)
        
        # Use FFmpeg to convert the audio format
        command = [
            "ffmpeg", "-y", "-i", input_file, "-acodec", "pcm_s16le", "-ar", "16000", output_file
        ]
        subprocess.run(command, check=True)

        with open(output_file, "rb") as f:
            converted_audio = f.read()
        
        return converted_audio
    except subprocess.CalledProcessError as e:
        print("Error during audio conversion:", str(e))
        return audio_binary  # Return original binary on error

def speech_to_text(audio_binary):

    # Convert audio to correct format
    audio_binary = convert_audio_format(audio_binary)

    # Set up Watson Speech-to-Text HTTP Api url
    base_url = 'https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/2bcce39d-4263-4835-829d-d0343331adf6'
    api_url = base_url+'/v1/recognize'

    # Set up parameters for our HTTP reqeust
    params = {
        'model': 'en-US_Multimedia',
    }

    # Set up headers including the API key
    headers = {
        'Authorization': 'Basic ' + base64.b64encode(b'apikey:iHpIBRurOD4OGYsFdUnDjvE8zCHmGGs9pyyBRYBmCWqE').decode('utf-8'),
        'Content-Type': 'audio/wav',
    }

    # Set up the body of our HTTP request
    body = audio_binary

    # Send a HTTP Post request
    response = requests.post(api_url, headers=headers, params=params, data=audio_binary).json()

    # Parse the response to get our transcribed text
    text = 'null'
    while bool(response.get('results')):
        print('Speech-to-Text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text: ', text)
        return text

def text_to_speech(text, voice=""):
    # Set up Watson Text-to-Speech HTTP Api url
    base_url = 'https://api.us-south.text-to-speech.watson.cloud.ibm.com/instances/85dc542f-a654-4b4e-82c5-fddf95448eb5'
    api_url = base_url + '/v1/synthesize'

    # Adding voice parameter in api_url if the user has selected a preferred voice
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    # Set the headers for our HTTP request
    headers = {
        'Authorization': 'Basic ' + base64.b64encode(b'apikey:mvUcHHKprvGju_v9-8V5Yvbr2nNAYTRdGK2TysQ6HLol').decode('utf-8'),
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    # Set the body of our HTTP request
    json_data = {
        'text': text,
    }

    # Send a HTTP Post reqeust to Watson Text-to-Speech Service
    response = requests.post(api_url, headers=headers, json=json_data)
    print('Text-to-Speech response:', response)
    return response.content

def watsonx_process_message(user_message):
    # Set the prompt for Watsonx API
    prompt = f"""You are an assistant helping translate sentences from English into Spanish.
    Translate the query to Spanish: ```{user_message}```."""
    response_text = model.generate_text(prompt=prompt)
    print("wastonx response:", response_text)
    return response_text


