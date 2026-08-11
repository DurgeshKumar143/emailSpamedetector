from django.contrib import admin
from .models import SpamPrediction

@admin.register(SpamPrediction)
class SpamPredictionAdmin(admin.ModelAdmin):
    """
    Admin configuration for SpamPrediction model.
    """
    list_display = ('id', 'prediction_label', 'confidence', 'created_at')
    list_filter = ('prediction_label', 'created_at')
    search_fields = ('email_text', 'prediction_label')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)
