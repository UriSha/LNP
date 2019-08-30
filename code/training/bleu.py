# -*- coding: utf-8 -*-
# Natural Language Toolkit: BLEU Score
#
# Copyright (C) 2001-2016 NLTK Project
# Authors: Chin Yee Lee, Hengfeng Li, Ruxin Hou, Calvin Tanujaya Lim
# Contributors: Dmitrijs Milajevs, Liling Tan
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

"""BLEU score implementation."""
from __future__ import division

import fractions
import math
import torch
from collections import Counter
from nltk.util import ngrams

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction


def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                  smoothing_function=None):
    """
    Calculate BLEU score (Bilingual Evaluation Understudy) from
    Papineni, Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.
    "BLEU: a method for automatic evaluation of machine translation."
    In Proceedings of ACL. http://www.aclweb.org/anthology/P02-1040.pdf
    :param references: reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)
    :return: The sentence-level BLEU score.
    :rtype: float
    """
    return corpus_bleu([references], [hypothesis], weights, smoothing_function)


def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=None):
    """
    Calculate a single corpus-level BLEU score (aka. system-level BLEU) for all
    the hypotheses and their respective references.
    Instead of averaging the sentence level BLEU scores (i.e. marco-average
    precision), the original BLEU metric (Papineni et al. 2002) accounts for
    the micro-average precision (i.e. summing the numerators and denominators
    for each hypothesis-reference(s) pairs before the division).
    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)
    :return: The corpus-level BLEU score.
    :rtype: float
    """
    # Before proceeding to compute BLEU, perform sanity checks.

    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(
        hypotheses), "The number of hypotheses and their reference(s) should be the same"

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0, 0

    # Smoothen the modified precision.
    # Note: smooth_precision() converts values into float.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    p_n = smoothing_function(p_n, references=references,
                             hypothesis=hypothesis, hyp_len=hyp_len)

    # Calculates the overall modified precision for all ngrams.
    # By sum of the product of the weights and the respective *p_n*
    s = (w * math.log(p_i) for w, p_i in zip(weights, p_n)
         if p_i.numerator != 0)

    # return bp * math.exp(math.fsum(s))
    bleu = math.exp(math.fsum(s))
    bleup = bleu * bp

    return bleu, bp * bleup


def modified_precision(references, hypothesis, n):
    """
    Calculate modified ngram precision.
    The normal precision method may lead to some wrong translations with
    high-precision, e.g., the translation, in which a word of reference
    repeats several times, has very high precision.
    This function only returns the Fraction object that contains the numerator
    and denominator necessary to calculate the corpus-level precision.
    To calculate the modified precision for a single pair of hypothesis and
    references, cast the Fraction object into a float.
    The famous "the the the ... " example shows that you can get BLEU precision
    by duplicating high frequency words.
    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: A hypothesis translation.
    :type hypothesis: list(str)
    :param n: The ngram order.
    :type n: int
    :return: BLEU's modified precision for the nth order ngram.
    :rtype: Fraction
    """
    # Extracts all ngrams in hypothesis.
    counts = Counter(ngrams(hypothesis, n))

    # Extract a union of references' counts.
    ## max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n))
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {ngram: min(count, max_counts[ngram])
                      for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)


def closest_ref_length(references, hyp_len):
    """
    This function finds the reference that is the closest length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)
    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: The length of the hypothesis.
    :type hypothesis: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    """
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
    (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len


def brevity_penalty(closest_ref_len, hyp_len):
    """
    Calculate brevity penalty.
    As the modified n-gram precision still has the problem from the short
    length sentence, brevity penalty is used to modify the overall BLEU
    score according to length.
    An example from the paper. There are three references with length 12, 15
    and 17. And a concise hypothesis of the length 12. The brevity penalty is 1.
    :param hyp_len: The length of the hypothesis for a single sentence OR the
    sum of all the hypotheses' lengths for a corpus
    :type hyp_len: int
    :param closest_ref_len: The length of the closest reference for a single
    hypothesis OR the sum of all the closest references for every hypotheses.
    :type closest_reference_len: int
    :return: BLEU's brevity penalty.
    :rtype: float
    """
    if hyp_len > closest_ref_len:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)


class SmoothingFunction:
    """
    This is an implementation of the smoothing techniques
    for segment-level BLEU scores that was presented in
    Boxing Chen and Collin Cherry (2014) A Systematic Comparison of
    Smoothing Techniques for Sentence-Level BLEU. In WMT14.
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """

    def __init__(self, epsilon=0.1, alpha=5, k=5):
        """
        This will initialize the parameters required for the various smoothing
        techniques, the default values are set to the numbers used in the
        experiments from Chen and Cherry (2014).
        :param epsilon: the epsilon value use in method 1
        :type epsilon: float
        :param alpha: the alpha value use in method 6
        :type alpha: int
        :param k: the k value use in method 4
        :type k: int
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k

    def method0(self, p_n, *args, **kwargs):
        """ No smoothing. """
        return p_n

    def method1(self, p_n, *args, **kwargs):
        """
        Smoothing method 1: Add *epsilon* counts to precision with 0 counts.
        """
        return [(p_i.numerator + self.epsilon) / p_i.denominator
                if p_i.numerator == 0 else p_i for p_i in p_n]

    def method2(self, p_n, *args, **kwargs):
        """
        Smoothing method 2: Add 1 to both numerator and denominator from
        Chin-Yew Lin and Franz Josef Och (2004) Automatic evaluation of
        machine translation quality using longest common subsequence and
        skip-bigram statistics. In ACL04.
        """
        return [Fraction(p_i.numerator + 1, p_i.denominator + 1, _normalize=False) for p_i in p_n]

    def method3(self, p_n, *args, **kwargs):
        """
        Smoothing method 3: NIST geometric sequence smoothing
        The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each
        precision score whose matching n-gram count is null.
        k is 1 for the first 'n' value for which the n-gram match count is null/
        For example, if the text contains:
         - one 2-gram match
         - and (consequently) two 1-gram matches
        the n-gram count for each individual precision score would be:
         - n=1  =>  prec_count = 2     (two unigrams)
         - n=2  =>  prec_count = 1     (one bigram)
         - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
         - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
        """
        incvnt = 1  # From the mteval-v13a.pl, it's referred to as k.
        for i, p_i in enumerate(p_n):
            if p_i.numerator == 0:
                p_n[i] = 1 / (2 ** incvnt * p_i.denominator)
                incvnt += 1
        return p_n

    def method4(self, p_n, references, hypothesis, hyp_len):
        """
        Smoothing method 4:
        Shorter translations may have inflated precision values due to having
        smaller denominators; therefore, we give them proportionally
        smaller smoothed counts. Instead of scaling to 1/(2^k), Chen and Cherry
        suggests dividing by 1/ln(len(T)), where T is the length of the translation.
        """
        incvnt = 1
        for i, p_i in enumerate(p_n):
            if p_i.numerator == 0 and hyp_len != 0:
                p_n[i] = incvnt * self.k / math.log(hyp_len)  # Note that this K is different from the K from NIST.
                incvnt += 1
        return p_n

    def method5(self, p_n, references, hypothesis, hyp_len):
        """
        Smoothing method 5:
        The matched counts for similar values of n should be similar. To a
        calculate the n-gram matched count, it averages the n−1, n and n+1 gram
        matched counts.
        """
        m = {}
        # Requires an precision value for an addition ngram order.
        p_n_plus1 = p_n + [modified_precision(references, hypothesis, 5)]
        m[-1] = p_n[0] + 1
        for i, p_i in enumerate(p_n):
            p_n[i] = (m[i - 1] + p_i + p_n_plus1[i + 1]) / 3
            m[i] = p_n[i]
        return p_n

    def method6(self, p_n, references, hypothesis, hyp_len):
        """
        Smoothing method 6:
        Interpolates the maximum likelihood estimate of the precision *p_n* with
        a prior estimate *pi0*. The prior is estimated by assuming that the ratio
        between pn and pn−1 will be the same as that between pn−1 and pn−2.
        """
        for i, p_i in enumerate(p_n):
            if i in [1, 2]:  # Skips the first 2 orders of ngrams.
                continue
            else:
                pi0 = 0 if p_n[i - 2] == 0 else p_n[i - 1] ** 2 / p_n[i - 2]
                # No. of ngrams in translation.
                l = sum(1 for _ in ngrams(hypothesis, i + 1))
                p_n[i] = (p_i + self.alpha * pi0) / (l + self.alpha)
        return p_n

    def method7(self, p_n, references, hypothesis, hyp_len):
        """
        Smoothing method 6:
        Interpolates the maximum likelihood estimate of the precision *p_n* with
        a prior estimate *pi0*. The prior is estimated by assuming that the ratio
        between pn and pn−1 will be the same as that between pn−1 and pn−2.
        """
        p_n = self.method4(p_n, references, hypothesis, hyp_len)
        p_n = self.method5(p_n, references, hypothesis, hyp_len)
        return p_n


def corpus_bleu_with_joint_refrences(joint_refs, list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                                     smoothing_function=None):

    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(
        hypotheses), "The number of hypotheses and their reference(s) should be the same"

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.

        refs_with_joint_refs = references.copy()
        refs_with_joint_refs.extend(joint_refs)

        for i, _ in enumerate(weights, start=1):

            p_i = modified_precision(refs_with_joint_refs, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(refs_with_joint_refs, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0, 0

    # Smoothen the modified precision.
    # Note: smooth_precision() converts values into float.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    p_n = smoothing_function(p_n, references=refs_with_joint_refs,
                             hypothesis=hypothesis, hyp_len=hyp_len)

    # Calculates the overall modified precision for all ngrams.
    # By sum of the product of the weights and the respective *p_n*
    s = (w * math.log(p_i) for w, p_i in zip(weights, p_n)
         if p_i.numerator != 0)

    # return bp * math.exp(math.fsum(s))
    bleu = math.exp(math.fsum(s))
    bleup = bleu * bp

    return bleu, bp * bleup


def populate_predicted_and_ground_truth(sent_ys, target_xs, predictions, id2w):

    predicted_sentences = []
    ground_truth_sentences = []
    for sent_y, target_x, prediction in zip(sent_ys, target_xs, predictions):
        masked_positions = {}
        for i, pos_tensor in enumerate(target_x):
            pos = pos_tensor.item()
            if pos >= len(sent_y):
                break
            masked_positions[pos] = i

        orig = []
        pred = []
        for i, word_id_tensor in enumerate(sent_y):
            word_id = word_id_tensor.item()
            if word_id == 0:
                break

            if i in masked_positions:
                pred_id = torch.max(prediction[masked_positions[i]], dim=0)[1].item()
                pred_id += 1  # account for padding shift
                pred.append(id2w[pred_id])
                orig.append(id2w[word_id])
            else:
                word = id2w[word_id]
                pred.append(word)
                orig.append(word)

        ground_truth_sentences.append([orig])
        predicted_sentences.append(pred)

    return predicted_sentences, ground_truth_sentences
