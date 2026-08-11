from django.db import models

class SpamPrediction(models.Model):
    """
    Model to store the spam email detection history logs.
    """
    email_text = models.TextField(verbose_name="Original Email Text")
    cleaned_text = models.TextField(verbose_name="Cleaned Preprocessed Text", blank=True, null=True)
    prediction_label = models.CharField(max_length=10, verbose_name="Prediction (Spam/Ham)")
    confidence = models.FloatField(verbose_name="Confidence Score (%)")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Prediction Date & Time")

    class Meta:
        ordering = ['-created_at'] # Show newest predictions first
        verbose_name = "Spam Prediction"
        verbose_name_plural = "Spam Predictions"

    def __str__(self):
        return f"{self.prediction_label} ({self.confidence:.2f}%) at {self.created_at.strftime('%Y-%m-%d %H:%M')}"
