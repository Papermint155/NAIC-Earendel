import concurrent.futures
from bing_image_downloader import downloader

def worker_thread(query_str: str, limit: int):
  downloader.download(
    query_str,
    limit=limit,
    output_dir='dataset',
    adult_filter_off=True,
    force_replace=False,
    timeout=60,
    verbose=True)

if __name__ == "__main__":
  #Add new search key into array ["cat","t-shirt","apple","orange","pear","fish"]
  query_strings = [
    "kuih talam",
    "kuih seri muka",
    "kuih ubi kayu",
    "kuih kaswi pandan",
    "kuih ketayap",
    "onde-onde",
    "kuih lapis",
    "kek lapis"
  ]

  #Parameters
  number_of_images = 200                  # Desired number of images
  number_of_workers = len(query_strings)//2 # Number of "workers" used

  #Run each query_str in a separate thread
  #Automatically waits for all threads to finish
  #Removes duplicate strings from query_strings
  with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executor:
    executor.map(worker_thread, query_strings, [number_of_images] * len(query_strings))
