# youtube-transcript-api       0.6.2

# get_transcript method will used in above  line


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# from youtube_transcript_api.errors import , NoTranscriptFound

# Modern Langchain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_community.vectorstores import FAISS
# from langchain_classic.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# video_id = "sVcwVQRHIc8"
video_id = "8aAItQeRlcc"

try:
    # 1. FETCH: The correct method is get_transcript()
    transcript_raw = YouTubeTranscriptApi.get_transcript(
        video_id="8aAItQeRlcc", languages=["en"])
    print(transcript_raw)
    # # 2. CLEAN: Combine the chunks into a single readable string
    transcript_text = " ".join(chunk["text"] for chunk in transcript_raw)

    print(f"Success! Fetched transcript with {len(transcript_text)} characters.\n")
    print(transcript_text[:500] + "...\n") # Printing just the first 500 chars to avoid console flood

except TranscriptsDisabled:
    print("Error: Transcripts are disabled for this video.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
