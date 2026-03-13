import numpy as np
import json
import os
import heapq
from sentence_transformers import SentenceTransformer
from transformers import logging
logging.set_verbosity_error()

class NanoVectorDB:
  def __init__(self, model_name='all-MiniLM-L6-v2'):
    self.model = SentenceTransformer(model_name)
    self.allchunks = []
    self.vectorized_chunks = []
    self.fileNames = []
    self._load_from_disk()
    
  def setup(self, content, fileName, chunksize = 500):
    if fileName not in self.fileNames:
      self.fileNames.append(fileName)
    
    overlap = int(0.15*chunksize)
    stepsize = chunksize - overlap
    text_chunks = []
    format_chunks = []

    for i in range(0,len(content), stepsize):
      chunk = content[i:i+chunksize]
      if i + chunksize < len(content):
        indexSpace = chunk.rfind(" ")
        if indexSpace != -1:
          chunk = chunk[0:indexSpace]
      text_chunks.append(chunk)
      format_chunks.append({"text":chunk, "source": fileName})

    self.allchunks.extend(format_chunks)

    if self.vectorized_chunks is None or self.vectorized_chunks.size == 0: 
      self.vectorized_chunks = self.model.encode(text_chunks)
    else:
      self.vectorized_chunks = np.vstack([self.vectorized_chunks, self.model.encode(text_chunks)])

    self.save_to_disk()

  def question(self, question):
    vectorized_question = np.array(self.model.encode(question))
    chunks = []
    scores = (self.vectorized_chunks @ vectorized_question)/(np.linalg.norm(vectorized_question)*np.linalg.norm(self.vectorized_chunks,axis=1))
    ans = []

    for i in self.fileNames:
      file_index = [idx for idx, chunk in enumerate(self.allchunks) if chunk["source"] == i]
      if not file_index:
        continue
      best_indexes = np.argsort(scores[file_index])[-3:][::-1]
      for top in best_indexes:
        if scores[file_index[top]] < 0.35:
          break
        ans.append(self.allchunks[file_index[top]])

    return ans
  
  def save_to_disk(self, save_dir= "db_storage"):
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    np.save(os.path.join(save_dir, "vectors.npy"), self.vectorized_chunks)

    with open(os.path.join(save_dir, "chunks.json"), "w") as f:
      json.dump({"chunks": self.allchunks, "fileNames": self.fileNames}, f)
        
  def _load_from_disk(self, save_dir="db_storage"):
    vector_file = os.path.join(save_dir, "vectors.npy")
    chunk_file = os.path.join(save_dir, "chunks.json")
    
    if os.path.exists(vector_file) and os.path.exists(chunk_file):
      self.vectorized_chunks = np.load(vector_file)
      with open(chunk_file, "r") as f:
        data = json.load(f)
        self.allchunks = data["chunks"]
        self.fileNames = data.get("fileNames", []) 
      print(f"Loaded {len(self.allchunks)} chunks from disk.")
    else:
      print("No existing database found. Starting fresh.") 