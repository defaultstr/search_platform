#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from .bing_api import BingSE
from .models import *
import json


def crawl_bing(query):
    results = json.loads(BingSE.BingSE(query.encode('utf8')))
    res_docs = []
    for r in results:
        r_doc = BingResult(
            title=r['title'],
            url=r['url'],
            snippet=r['snippet'],
            html_content=r['html'],
        )
        res_docs.append(r_doc)
    results_doc = BingResults(query=query, results=res_docs)
    results_doc.save()
    return BingResults.objects.get(query=query)


