"""Sanity checks for the bundled dataset loaders."""
from __future__ import annotations

from blockwise import datasets


def test_load_bike_has_response():
    df = datasets.load_bike()
    assert len(df) > 10_000
    assert "cnt" in df.columns


def test_load_adult_has_binary_response():
    df = datasets.load_adult()
    assert len(df) > 10_000
    assert "salary" in df.columns
    # After preprocess, salary should be 0/1.
    assert set(df["salary"].unique()).issubset({0, 1})


def test_load_house_has_price():
    df = datasets.load_house()
    assert len(df) > 10_000
    assert "price" in df.columns


def test_load_house_sampling():
    df = datasets.load_house(n_sample=200, seed=0)
    assert len(df) == 200
