# Manual of the Scripts Here

- 2021-02-03

----

## Generate Abstract Summary of astro-ph Pre-prints

- Use the script: `astroph_abstract.py`
    - This is inspired and heavily based on [`arxivscraper`](https://github.com/Mahdisadjadi/arxivscraper) by [Mahdi Sadjadi](https://github.com/Mahdisadjadi)

### Requirements

- `numpy`, `astropy`
- If you wish to use the automatic sentence break feature, you need to install the `nltk` (natural language toolkit) package, and download some basic dataset.
    - First, in command line: `pip install --upgrade nltk`
    - Then in Python: `import nltk; nltk.download('popular')`
    - If you want to download the whole dataset, you can use `nltk.download()`, then select the dataset from the GUI tool. The total dataset is ~4Gb.

### Basic Usage

- Show help message:
    - `python3 astroph_abstract.py -h`

- Get today's astro-ph pre-prints and output to a Markdown file named `today.md`:
    - `python3 astroph_abstract.py -t today -o today.md`

- Get yesterday's list and only keeps the astroph.GA (Galaxy and Extragalactic) ones in the default `output.md` file:
    - `python3 astroph_abstract.py -t yesterday -s GA`

- Get the GA and CO (Cosmology) items from the last seven days including the cross-listed items:
    - **Note**: There are a lot of cross-listed cosmology preprints from the physics side. Be careful!
    - `python3 astroph_abstract.py -t past_seven -s GA CO -c`

- Search the IM (Instruments and Methods) listing from 2021-01-01 to 2021-01-30, show progress:
    - `python3 astroph_abstract.py -f 2021-01-01 -u 2021-01-30 -s IM -v`

### In `Python`

- You can access the same function by using 
    - `from astroph_abstract import astroph_abstract`
    - The function also returns an `astropy.table` object that includes all the records from the search.

### Problems:

- Current search is using the `updated` key in the metadata. arXiv Open Archive API is not flexible in term of the types of date we can search for. However, there could be many old pre-prints that have new updates. To exclude these old pre-prints, we rely on the `created` date keyword in the metadata. But it is still different from the date of appearance on the arXiv website. 
    - For example, one could create a new pre-print on the weekend, but it will only show up on the website on Monday. 
    - To deal with this issue, we setup a "cushion" before the `date_from` keyword (the beginning date of the search). The default value is 2.5 (two-and-half days).
    - But this doesn't solve all issues: sometimes the authors immediately update the pre-print one day after the initial submission. In that case, the preprint will be selected in the search result for the following day.
