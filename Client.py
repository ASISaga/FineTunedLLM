# Example usage of the AbstractiveSummarizer class
from KnowledgeModel.AbstractiveSummarizer import AbstractiveSummarizer

def main():
    """
    Main function to demonstrate the usage of the AbstractiveSummarizer class.
    """
    # Define the input text to be summarized.
    input_text = """
    Artificial intelligence (AI) is rapidly evolving, and one of its most exciting applications
    is in the field of natural language processing. Modern AI systems can understand, generate, 
    and even translate text with impressive accuracy. Among these systems, transformers have 
    revolutionized how we approach tasks like summarization, question answering, and language translation.
    Abstractive summarization, in particular, leverages these advanced models to generate summaries 
    that are not merely copies of the original content but are rephrased, concise, and often more 
    coherent in conveying the underlying meaning of the text.
    """

    # Create an instance of the AbstractiveSummarizer class.
    summarizer = AbstractiveSummarizer()

    # Generate the summary.
    summary = summarizer.summarize_text(
        input_text,
        max_length=100,
        min_length=40,
        do_sample=False
    )

    # Print the generated summary.
    print("Summary:", summary)

if __name__ == "__main__":
    main()