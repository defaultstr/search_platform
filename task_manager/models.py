#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from mongoengine import *
from user_system.models import User

try:
    import simplejson as json
except ImportError:
    import json


