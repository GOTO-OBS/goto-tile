from django.conf.urls import url
from .views import SkyMapView


app_name = 'webtile'

urlpatterns = [
    url(r'^$',
        SkyMapView.as_view(),
        name='main'),
]
