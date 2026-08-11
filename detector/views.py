from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Avg

from .models import SpamPrediction
from .forms import EmailInputForm, ContactForm
from .services import predict_email

def home(request):
    """
    Renders the dashboard with key system metrics: total scans, spam counts,
    and average prediction confidence.
    """
    total_scans = SpamPrediction.objects.count()
    spam_count = SpamPrediction.objects.filter(prediction_label='Spam').count()
    ham_count = SpamPrediction.objects.filter(prediction_label='Ham').count()
    
    # Calculate percentages
    spam_pct = round((spam_count / total_scans * 100), 1) if total_scans > 0 else 0
    ham_pct = round((ham_count / total_scans * 100), 1) if total_scans > 0 else 0
    
    # Calculate average confidence
    avg_confidence_data = SpamPrediction.objects.aggregate(Avg('confidence'))
    avg_confidence = round(avg_confidence_data['confidence__avg'] or 0, 2)
    
    # Fetch the 5 most recent predictions
    recent_predictions = SpamPrediction.objects.all()[:5]

    context = {
        'total_scans': total_scans,
        'spam_count': spam_count,
        'ham_count': ham_count,
        'spam_pct': spam_pct,
        'ham_pct': ham_pct,
        'avg_confidence': avg_confidence,
        'recent_predictions': recent_predictions,
        'active_page': 'home'
    }
    return render(request, 'detector/home.html', context)


def detect_spam(request):
    """
    Handles email classification requests. Processes the input, gets the prediction
    from our machine learning model, and saves the outcome in the database.
    """
    result = None
    form = EmailInputForm()

    if request.method == 'POST':
        form = EmailInputForm(request.POST)
        if form.is_valid():
            email_text = form.cleaned_data['email_text']
            
            # Call our prediction service
            prediction_res = predict_email(email_text)
            
            if prediction_res['error']:
                messages.error(request, prediction_res['error'])
            else:
                result = prediction_res
                
                # Save predictions into the database for logging and analytics
                db_record = SpamPrediction.objects.create(
                    email_text=email_text,
                    cleaned_text=prediction_res['cleaned_text'],
                    prediction_label=prediction_res['prediction_label'],
                    confidence=prediction_res['confidence']
                )
                messages.success(request, "Analysis completed successfully!")
                
    context = {
        'form': form,
        'result': result,
        'active_page': 'detect'
    }
    return render(request, 'detector/detect.html', context)


def prediction_history(request):
    """
    Displays historical logs of all spam queries, with search capability and pagination.
    """
    search_query = request.GET.get('search', '')
    
    # Apply text filter if a search query is provided
    if search_query:
        predictions_list = SpamPrediction.objects.filter(
            email_text__icontains=search_query
        )
    else:
        predictions_list = SpamPrediction.objects.all()

    # Paginate predictions (10 records per page)
    paginator = Paginator(predictions_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'active_page': 'history'
    }
    return render(request, 'detector/history.html', context)


def about(request):
    """
    Educational details about the project. Explains preprocessing pipeline and ML models.
    """
    context = {
        'active_page': 'about'
    }
    return render(request, 'detector/about.html', context)


def contact(request):
    """
    Renders the contact page and processes contact form submissions.
    """
    form = ContactForm()
    
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # In a real application, you would send an email here.
            # For this educational project, we will show a success notification.
            name = form.cleaned_data['name']
            messages.success(request, f"Thank you, {name}! Your message has been received.")
            return redirect('contact')
            
    context = {
        'form': form,
        'active_page': 'contact'
    }
    return render(request, 'detector/contact.html', context)
