from django import forms

class EmailInputForm(forms.Form):
    """
    Form to accept input email/message text for spam analysis.
    """
    email_text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 6,
            'placeholder': 'Paste the content of your email or SMS message here...',
            'style': 'resize: none; border-radius: 10px;',
            'id': 'email_text_input'
        }),
        label="Email Content",
        max_length=10000,
        required=True
    )


class ContactForm(forms.Form):
    """
    Form for the contact page of the web application.
    """
    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your full name',
            'style': 'border-radius: 8px;',
            'id': 'contact_name'
        })
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'style': 'border-radius: 8px;',
            'id': 'contact_email'
        })
    )
    subject = forms.CharField(
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter message subject',
            'style': 'border-radius: 8px;',
            'id': 'contact_subject'
        })
    )
    message = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 4,
            'placeholder': 'Write your message here...',
            'style': 'resize: none; border-radius: 8px;',
            'id': 'contact_message'
        })
    )
