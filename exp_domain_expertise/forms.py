#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from django import forms


class PreTaskQuestionForm(forms.Form):
    input_description = forms.CharField(required=True, min_length=1)
    knowledge_scale = forms.IntegerField(required=True, min_value=1, max_value=5)
    interest_scale = forms.IntegerField(required=True, min_value=1, max_value=5)
    difficulty_scale = forms.IntegerField(required=True, min_value=1, max_value=5)
