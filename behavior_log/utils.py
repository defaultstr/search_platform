#!/user/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'defaultstr'

from mongoengine import *
from mongoengine.common import _import_class
from bson import ObjectId
from datetime import datetime
from user_system.models import User
import json


def parse_val(x):
    if isinstance(x, str) or isinstance(x, unicode):
        try:
            return json.loads(x)
        except ValueError:
            return x
    else:
        return x


def parse_list(l, add_type=True, remove_object_id=True):
    res = []
    for x in l:
        if isinstance(x, list):
            res.append(parse_list(x, add_type=add_type, remove_object_id=remove_object_id))
        elif isinstance(x, Document):
            res.append(document_to_dict(x, add_type=True, remove_object_id=remove_object_id))
        else:
            res.append(parse_val(x))
    return res


def document_to_dict(doc, add_type=True, remove_object_id=True):
    assert(isinstance(doc, Document))
    DeReference = _import_class('DeReference')
    doc_dict = DeReference()(doc._data, 5)
    res = {}
    if add_type:
        res['type'] = doc.__class__.__name__
    for key in doc:
        val = doc_dict[key]

        if isinstance(val, Document):
            # reference field
            if isinstance(val, User):
                res[key] = {'type': 'User', 'username': val.username}
            else:
                res[key] = document_to_dict(val, add_type=add_type, remove_object_id=remove_object_id)
        elif isinstance(val, ObjectId):
            # object id
            if not remove_object_id:
                res[key] = str(val)
        elif isinstance(val, list):
            # list field
            res[key] = parse_list(val, add_type=add_type, remove_object_id=remove_object_id)
        else:
            # normal field
            res[key] = parse_val(val)

    return res


def json_default(obj):
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')


def document_to_json(doc, prettify=False, indent=0, ensure_ascii=False, add_type=True, remove_object_id=True):
    assert(isinstance(doc, Document))
    return json.dumps(
        document_to_dict(doc, add_type=add_type, remove_object_id=remove_object_id),
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=json_default,
    )


def save_list_to_db(l):
    assert(isinstance(l, list))
    for e in l:
        if isinstance(e, Document):
            save_doc_to_db(e)
        elif isinstance(e, list):
            save_list_to_db(e)


def save_doc_to_db(doc):
    assert(isinstance(doc, Document))
    doc_dict = doc._data
    for key in doc:
        val = doc_dict[key]
        if isinstance(val, Document):
            # reference field
            save_doc_to_db(val)
        elif isinstance(val, list):
            # list field
            save_list_to_db(val)
    doc.save()


def output_mhtml(mhtml, filename):
    import base64
    data = base64.b64decode(mhtml[13:])
    with open(filename, 'wb') as fout:
        print >>fout, data




