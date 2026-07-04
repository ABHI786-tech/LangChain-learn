from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

"""N*N full course video by Nick saraev"""
video_id = "2GZ2SNXWK-c"
try: 
  youtube = YouTubeTranscriptApi()
  transcript_list = youtube.fetch(video_id, languages=(["en"]))
  # print(transcript_list)
  transcript = " ".join(chunk.text for chunk in transcript_list)
  print("___________________________",transcript,"___________________________________")
  
except TranscriptsDisabled:
  print("no caption available")