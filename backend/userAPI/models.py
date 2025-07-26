from django.db import models

class Log(models.Model):
    date = models.DateField()
    location = models.CharField(max_length=255)
    level = models.IntegerField()
    details = models.TextField()

    def __str__(self):
        return f"{self.date} - {self.location} - {self.level}"