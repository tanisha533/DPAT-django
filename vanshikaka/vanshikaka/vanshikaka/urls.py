from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render

def home(request):
    return render(request, 'application/home.html')

def index(request):
    return render(request, 'application/index.html')

urlpatterns = [
    path('', home, name='home'),
    path('index/', index, name='index'),
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
] 