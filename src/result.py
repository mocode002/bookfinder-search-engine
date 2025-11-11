def show_search_results(top_indices, similarities, books_df, top_k=5):
    """Print simple text-based search results (for terminal use)."""
    for rank, idx in enumerate(top_indices[:top_k]):
        row = books_df.iloc[idx]
        title = row.get('title', 'Unknown Title')
        similarity = similarities[rank]
        description = row.get('description', '')
        genres = ", ".join(row['genres']) if isinstance(row.get('genres'), list) else row.get('genres', 'N/A')

        print(f"📘 Rank {rank+1} | Similarity: {similarity:.4f}")
        print(f"Title: {title}")
        print(f"Genres: {genres}")
        print(f"Description: {description[:200]}...")
        print("-" * 90)
