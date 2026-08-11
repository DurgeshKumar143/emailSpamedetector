from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detect/', views.detect_spam, name='detect'),
    path('history/', views.prediction_history, name='history'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
]
