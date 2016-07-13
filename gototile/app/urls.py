from django.conf.urls import url
from .views import SkyMapView


urlpatterns = [
    url(r'^$',
        SkyMapView.as_view(),
        name='gototile-skymap'),
]
