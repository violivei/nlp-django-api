# coding=utf-8
from api.models import Task
from django.http import Http404

from api.serializers import TaskSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from django.http import JsonResponse
import conv_net_sentence


class GetClass(APIView):
    """
    List all tasks, or create a new task.
    """
    def get(self, request, format=None):
        final_class = conv_net_sentence.get_predict("mr.p", "question: Linux training setup; excerpt: I want to give a tutorial/training type seminar on Linux. I also want to demonstrate certain Linux specific software with audience participation (on their own machines). What is the best way to go")
        data = {'predicted_class': final_class}
        return JsonResponse(data)

    def post(self, request, format=None):
        final_class = conv_net_sentence.get_predict("mr.p", request.data["description"])
        data = {'predicted_class': final_class}
        return JsonResponse(data)

    def delete(self, request, pk, format=None):
        task = self.get_object(pk)
        task.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class GetProject(APIView):
    """
    Retrieve, update or delete a task instance.
    """
    def get_object(self, pk):
        try:
            return Task.objects.get(pk=pk)
        except Task.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        final_class = conv_net_sentence.get_predict("modelproject.non-static", "Mensagem de input do serviço SubscriberControllerEBF")
        data = {'predicted_class': final_class}
        return JsonResponse(data)

    def post(self, request, format=None):
        final_class = conv_net_sentence.get_predict("modelproject.non-static", request.data["description"])
        data = {'predicted_class': final_class}
        return JsonResponse(data)

    def put(self, request, pk, format=None):
        task = self.get_object(pk)
        serializer = TaskSerializer(task, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        task = self.get_object(pk)
        task.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)