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
    self._load_from_disk()
    
  def setup(self, content, chunk_size = 200):
    words = content.split() 
    new_chunks = [] 
    overlap = int(chunk_size * 0.1)
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
      chunk_text = " ".join(words[i : i + chunk_size])
      self.allchunks.append(chunk_text)
      new_chunks.append(chunk_text)
    
    if len(self.vectorized_chunks) == 0:
      self.vectorized_chunks = self.model.encode(new_chunks)
    else:
      self.vectorized_chunks = np.vstack([self.vectorized_chunks, self.model.encode(new_chunks)])
    self.save_to_disk()

  def question(self, question, retrieve = 5):
    self.vectorized_np = np.array(self.vectorized_chunks) 
    self.vectorized = np.array(self.model.encode(question))
    chunks = []

    scores = (self.vectorized_np @ self.vectorized)/(np.linalg.norm(self.vectorized)*np.linalg.norm(self.vectorized_np,axis=1)) 
    best_indexes = np.argsort(scores)[-retrieve:][::-1]
    ans = []
    for i in best_indexes:
      ans.append(self.allchunks[i])
    return ans
  
  def save_to_disk(self, save_dir= "db_storage"):
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "vectors.npy"), self.vectorized_chunks)
    with open(os.path.join(save_dir, "chunks.json"), "w") as f:
      json.dump(self.allchunks, f)
        
  def _load_from_disk(self, save_dir="db_storage"):
    vector_file = os.path.join(save_dir, "vectors.npy")
    chunk_file = os.path.join(save_dir, "chunks.json")

    if os.path.exists(vector_file) and os.path.exists(chunk_file):
      self.vectorized_chunks = np.load(vector_file)
        
      with open(chunk_file, "r") as f:
        self.allchunks = json.load(f)
            
      print(f"Loaded {len(self.allchunks)} chunks from disk.")
    else:
      print("No existing database found. Starting fresh.")