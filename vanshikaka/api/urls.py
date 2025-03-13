from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard1/', views.dashboard1, name='dashboard1'),
    path('dashboard2/', views.dashboard2, name='dashboard2'),
    path('e_waste_chart/', views.e_waste_chart, name='e_waste_chart'),
    path('e_waste_data/', views.get_e_waste_data, name='e_waste_data'),
    path('e_waste_dataset_cleaned/', views.get_e_waste_data, name='get_e_waste_data'),
    path('co2_data/', views.get_co2_data, name='co2_data'),
    path('co2_kpis/', views.get_co2_kpis, name='co2_kpis'),
    path('co2_analytics/', views.process_co2_analytics, name='co2_analytics'),
    path('co2_visualizations/', views.get_co2_visualizations, name='co2_visualizations'),
    path('forecast/', views.forecast_e_waste, name='forecast_e_waste'),
    path('forecast_data/', views.get_forecast_data, name='forecast_data'),
    path('chat/', views.chat, name='chat'),
    path('strategies/', views.strategies, name='strategies'),
]