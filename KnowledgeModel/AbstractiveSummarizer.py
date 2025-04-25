class AbstractiveSummarizer:
    """
    A class to perform abstractive summarization using the transformers library.
    """

    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize the summarization pipeline with the specified model.

        :param model_name: The name of the pre-trained model to use for summarization.
        """
        from transformers import pipeline
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize_text(self, input_text, max_length=100, min_length=40, do_sample=False):
        """
        Generate a summary for the given input text.

        :param input_text: The text to be summarized.
        :param max_length: The maximum length of the summary.
        :param min_length: The minimum length of the summary.
        :param do_sample: Whether to use sampling for generating the summary.
        :return: The generated summary as a string.
        """
        summary = self.summarizer(
            input_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample
        )
        return summary[0]['summary_text']