def split_into_chunks(lst, chunk_size=4):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]