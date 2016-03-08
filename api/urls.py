from django.conf.urls import patterns, include, url
from django.contrib import admin

from api import views

admin.autodiscover()

urlpatterns = patterns('',
    url(r'^priority/', views.GetPriority.as_view()),
    url(r'^project/', views.GetProject.as_view()),
   # url(r'^tasks/(?P<pk>[0-9]+)/$', views.TaskDetail.as_view()),
)