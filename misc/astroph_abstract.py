#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate summary of astro-ph output."""

import time
import argparse
import datetime

import xml.etree.ElementTree as ET

from urllib.request import urlopen
from urllib.error import HTTPError

import numpy as np
from astropy.table import Table

try:
    from nltk import tokenize
    use_nltk = True
except ImportError:
    use_nltk = False

OAI = '{http://www.openarchives.org/OAI/2.0/}'
ARXIV = '{http://arxiv.org/OAI/arXiv/}'
BASE = 'http://export.arxiv.org/oai2?verb=ListRecords&'

ABS_URL = "http://arxiv.org/abs/{:s}"
PDF_URL = "http://arxiv.org/pdf/{:s}.pdf"

# By default, this only works for astro-ph.
CAT = "physics:astro-ph"
SUBCAT = ['GA', 'CO', 'EP', 'HE', 'IM', 'SR']

SEARCH_TYPE = ['today', 'yesterday', 'from_yesterday', 'past_seven', 'user']

TODAY = datetime.datetime.today().replace(hour=0, minute=0, second=0)
YESTERDAY = (TODAY - datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0)


def _filter_sub_class(papers, sub_cat, no_crosslist=True):
    """
    Filter the search results and keep the ones in certain sub-category.
    """
    if no_crosslist:
        # The first sub-category needs to be the desired one
        return [
            "astro-ph.{:s}".format(sub_cat.strip()) == p['sub_cat'].split()[0].strip()
            for p in papers]
    return ["astro-ph.{:s}".format(sub_cat.strip()) in p['sub_cat'] for p in papers]

def _get_text(meta, tag):
    """Extracts text from an xml field"""
    try:
        return meta.find(ARXIV + tag).text.strip().replace('\n', ' ')
    except:
        return ''

def _date_str(date):
    """
    Convert the datetime into a string with '%Y-%m-%d' format.
    """
    return date.strftime('%Y-%m-%d')


def gather_dates(search_type, date_from=None, date_until=None):
    """
    Get the from and until dates for the search.
    """
    if search_type is None:
        search_type = 'user'

    search_type = search_type.lower().strip()

    if search_type not in SEARCH_TYPE:
        raise ValueError(
            "Wrong search type: ", SEARCH_TYPE)

    if search_type == 'today':
        date_from = TODAY
        date_until = TODAY
    elif search_type == 'yesterday':
        date_from = YESTERDAY
        date_until = YESTERDAY
    elif search_type == 'from_yesterday':
        date_from = YESTERDAY
        date_until = TODAY
    elif search_type == 'past_seven':
        date_from = TODAY - datetime.timedelta(days=7)
        date_until = TODAY
    else:
        if date_from is None:
            date_from = TODAY
        else:
            try:
                date_from = datetime.datetime.strptime(date_from, '%Y-%m-%d')
            except ValueError:
                print("Date format should be: YYYY-MM-DD")

        if date_until is None:
            date_until = TODAY
        else:
            try:
                date_until = datetime.datetime.strptime(date_until, '%Y-%m-%d')
            except ValueError:
                print("Date format should be: YYYY-MM-DD")

    return date_from, date_until


def organize_meta(meta):
    """
    Organize the metadata.
    """
    return {
        'id': _get_text(meta, 'id'),
        'title': _get_text(meta, 'title'),
        'abstract': _get_text(meta, 'abstract'),
        'sub_cat': _get_text(meta, 'categories'),
        'created': datetime.datetime.strptime(_get_text(meta, 'created'), '%Y-%m-%d')
    }


def scrape(url, sleep_time=30, timeout=300, verbose=True):
    """
    Get the search results.
    """
    t0, elapsed = time.time(), 0
    results, batch = [], 1

    while True:
        if verbose:
            print('Fetching up to {:d} records...'.format(1000 * batch))

        # Arxiv only allows you to scrape 1000 words at a time.
        try:
            response = urlopen(url)
        except HTTPError as e:
            if e.code == 503:
                _ = int(e.hdrs.get('retry-after', sleep_time))
                print('Got 503. Retrying after {0:d} seconds.'.format(sleep_time))
                time.sleep(sleep_time)
                continue
            else:
                raise

        batch += 1

        # Get the full XML output
        xml_output = response.read()
        xml_root = ET.fromstring(xml_output)

        # Get all the search records
        records = xml_root.findall(OAI + 'ListRecords/' + OAI + 'record')

        for record in records:
            # Get the metadata of the record
            meta = record.find(OAI + 'metadata').find(ARXIV + 'arXiv')
            results.append(meta)

        try:
            token = xml_root.find(OAI + 'ListRecords').find(OAI + 'resumptionToken')
        except:
            return 1
        if token is None or token.text is None:
            break
        else:
            url = BASE + 'resumptionToken=%s' % token.text

        t1 = time.time()
        elapsed += (t1 - t0)

        if elapsed >= timeout:
            break
        else:
            t0 = time.time()

    if verbose:
        print('Total number of records {:d}'.format(len(results)))

    return results

def astroph_abstract(output='output.md', search_type='user', date_cushion=2.5,
                     date_from=None, date_until=None, sub_cat=None,
                     verbose=False, sleep_time=30, timeout=300, no_crosslist=True):
    """
    Gather the abstracts of the astro-ph within a period of time, and output a summary
    markdown file.

    Based on: https://github.com/Mahdisadjadi/arxivscraper by Mahdisadjadi
    """
    # Get the from and unitl date
    date_f, date_u = gather_dates(search_type, date_from=date_from, date_until=date_until)

    # Form the search URL
    search_url = BASE + 'from={:s}&until={:s}&metadataPrefix=arXiv&set={:s}'.format(
        _date_str(date_f).strip(), _date_str(date_u).strip(), CAT)

    metadata = scrape(search_url, sleep_time=sleep_time, timeout=timeout, verbose=verbose)

    paper_records = [organize_meta(meta) for meta in metadata]

    # Remove the recently updated one
    # TODO: This is not perfect
    # - If someone created a preprint long before the submission, it will be left out
    papers = Table(
        [p for p in paper_records if p['created'] >= (
            date_f - datetime.timedelta(days=date_cushion))])
    
    if sub_cat is not None:
        sub_str = " ".join(sub_cat)
        if verbose:
            print("Will only keep items from sub-category: [{:s}]".format(sub_str))
            if no_crosslist:
                print("Will exclude cross-listed items")
            else:
                print("Will include cross-listed items")
    else:
        sub_str = ""
    # Filter the search results through sub-categories
    if isinstance(sub_cat, str):
        papers_keep = papers[_filter_sub_class(papers, sub_cat, no_crosslist=no_crosslist)]
    elif isinstance(sub_cat, list):
        papers_keep = papers[np.logical_or.reduce(
            [_filter_sub_class(papers, s, no_crosslist=no_crosslist) for s in sub_cat])]
    else:
        papers_keep = papers
    
    if verbose:
        print("Keep {:d} preprints".format(len(papers_keep)))

    # Organize the results into markdown format (line-by-line)
    markdown_list = []


    if date_f == date_u:
        markdown_list.append("### {:s} [{:s}]".format(_date_str(date_f), sub_str))
    else:
        markdown_list.append("### {:s} to {:s} [{:s}]".format(
            _date_str(date_f), _date_str(date_u), sub_str))

    for p in papers_keep:
        abs_url = ABS_URL.format(p['id'])
        pdf_url = PDF_URL.format(p['id'])
        markdown_list.append(
            "\n##### [{:s}]({:s}) [(PDF)]({:s})\n".format(
                ' '.join(p['title'].split()), abs_url, pdf_url))

        abstract = p['abstract'].replace('\\,', ' ')
        abstract.replace('et al.', 'et al')
        if not use_nltk:
            markdown_list.append("- {:s}".format(' '.join(abstract.split())))
        else:
            sentences = tokenize.sent_tokenize(' '.join(abstract.split()))
            for s in sentences:
                markdown_list.append("- {:s}".format(s))

    # Write the markdown to file
    with open(output, 'w') as f:
        for line in markdown_list:
            f.write("{:s}\n".format(line))

    return papers_keep


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output', dest='output',
        help="Output markdown file name.",
        default='output.md')
    parser.add_argument(
        '-t', '--type', dest='search_type',
        help="Type of the search: ['today', 'yesterday', 'from_yesterday', 'past_seven', 'user']",
        default='user')
    parser.add_argument(
        '-f', '--from', dest='date_from',
        help="Search from this date. In YYYY-MM-DD format.",
        default=None)
    parser.add_argument(
        '-u', '--until', dest='date_until',
        help="Search until this date. In YYYY-MM-DD format.",
        default=None)
    parser.add_argument(
        '-s', '--sub_cat', dest='sub_cat', nargs='+',
        help="Sub-category to include: [GA, CO, HE, IM, SR, EP]; Can select more than one.",
        default=None)
    parser.add_argument(
        '-d', '--date_cushion', dest='date_cushion', type=float,
        help="The preprint can be created these days before the `date_from` date.",
        default=2.5)
    parser.add_argument(
        '-c', '--crosslist', action="store_false", dest='no_crosslist', 
        help="Including cross-list items.",
        default=True)
    parser.add_argument(
        '-v', '--verbose', action="store_true", dest='verbose', 
        help="Print the progress.", default=False)

    args = parser.parse_args()

    _ = astroph_abstract(
        output=args.output, search_type=args.search_type,
        date_cushion=args.date_cushion, date_from=args.date_from, date_until=args.date_until,
        sub_cat=args.sub_cat, verbose=args.verbose, sleep_time=30, timeout=300,
        no_crosslist=args.no_crosslist)
