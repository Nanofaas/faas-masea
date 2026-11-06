"""Selective requests carry only MADEA's (i,j,f), with bounded local retries."""

from copy import deepcopy

import json

from pathlib import Path

from types import SimpleNamespace

import numpy as np

import pandas as pd

import pytest

from utils import step_time

from utils.selective_memory import SelectiveMemory

def test_sender_selects_one_neighbor_and_retries_without_extra_fields():
  selector = SelectiveMemory()
  bids = pd.DataFrame({'i': [0, 0], 'j': [1, 2], 'f': [0, 0]})
  data = {None: {'beta': {(1, 2, 1): 8., (1, 3, 1): 5.}}}  # only sender's beta
  omega = np.array([[4.], [0.], [0.]])
  prices = np.zeros((3, 1))
  latency = np.zeros((3, 3))
  for expected, pending in [(1, True), (2, False), (None, False)]:
    got = selector.select(bids, omega, prices, data, latency, {'latency_weight': 0.})
    assert list(got.columns) == ['i', 'j', 'f']
    assert got.j.tolist() == ([] if expected is None else [expected])
    assert selector.pending == pending
    assert selector.retry(0, 2, 0., 1.) == pending
    assert not selector.retry(1, 2, 0., 1.)
    assert not selector.retry(0, 2, 1., 1.)
  # Successful progress permits another request; no global state leaks between steps.
  omega[0, 0] = 2
  assert selector.select(bids, omega, prices, data, latency, {}).j.tolist() == [1]
  assert SelectiveMemory().select(bids, omega, prices, data, latency, {}).j.tolist() == [1]
  # Prices and latency, already visible to the sender, can change its preference.
  prices[1, 0] = 4
  assert SelectiveMemory().select(bids, omega, prices, data, latency, {}).j.tolist() == [2]
  omega[0, 0] = 0
  assert selector.select(bids, omega, prices, data, latency, {}).empty
  assert not selector.pending

def test_satisfied_buyer_can_retry_after_its_accepted_flow_is_displaced():
  selector = SelectiveMemory()
  bids = pd.DataFrame({'i': [0], 'j': [1], 'f': [0]})
  data = {None: {'beta': {(1, 2, 1): 1.}}}
  omega, prices, latency = np.array([[4.], [0.]]), np.zeros((2, 1)), np.zeros((2, 2))
  assert selector.select(bids, omega, prices, data, latency, {}).j.tolist() == [1]
  # Confirmation removes this buyer's whole group once its demand is satisfied.
  omega[0, 0] = 0
  assert selector.select(bids.iloc[:0], omega, prices, data, latency, {}).empty
  omega[0, 0] = 4
  assert selector.select(bids, omega, prices, data, latency, {}).j.tolist() == [1]
