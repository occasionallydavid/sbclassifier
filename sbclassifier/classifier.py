# An implementation of a Bayes-like spam classifier.
#
# Paul Graham's original description:
#
#     http://www.paulgraham.com/spam.html
#
# A highly fiddled version of that can be retrieved from our CVS repository,
# via tag Last-Graham.  This made many demonstrated improvements in error
# rates over Paul's original description.
#
# This code implements Gary Robinson's suggestions, the core of which are
# well explained on his webpage:
#
#    http://radio.weblogs.com/0101454/stories/2002/09/16/spamDetection.html
#
# This is theoretically cleaner, and in testing has performed at least as
# well as our highly tuned Graham scheme did, often slightly better, and
# sometimes much better.  It also has "a middle ground", which people like:
# the scores under Paul's scheme were almost always very near 0 or very near
# 1, whether or not the classification was correct.  The false positives
# and false negatives under Gary's basic scheme (use_gary_combining) generally
# score in a narrow range around the corpus's best spam_cutoff value.
# However, it doesn't appear possible to guess the best spam_cutoff value in
# advance, and it's touchy.
#
# The last version of the Gary-combining scheme can be retrieved from our
# CVS repository via tag Last-Gary.
#
# The chi-combining scheme used by default here gets closer to the theoretical
# basis of Gary's combining scheme, and does give extreme scores, but also
# has a very useful middle ground (small # of msgs spread across a large range
# of scores, and good cutoff values aren't touchy).
#
# This implementation is due to Tim Peters et alia.

import math

from sbclassifier.chi2 import chi2Q


LN2 = math.log(2)

HAM_COUNT = 'nham'
SPAM_COUNT = 'nspam'
HAM_PREFIX = 'h:'
SPAM_PREFIX = 's:'


class Classifier:

    unknown_token_prob = 0.5
    unknown_token_strength = 0.45
    minimum_prob_strength = 0.1
    ham_cutoff = 0.2
    spam_cutoff = 0.9
    max_discriminators = 150

    def __init__(self, store):
        self.store = store

    # spamprob using the chi-squared implementation
    # Across vectors of length n, containing random uniformly-distributed
    # probabilities, -2*sum(ln(p_i)) follows the chi-squared distribution
    # with 2*n degrees of freedom.  This has been proven (in some
    # appropriate sense) to be the most sensitive possible test for
    # rejecting the hypothesis that a vector of probabilities is uniformly
    # distributed.  Gary Robinson's original scheme was monotonic *with*
    # this test, but skipped the details.  Turns out that getting closer
    # to the theoretical roots gives a much sharper classification, with
    # a very small (in # of msgs), but also very broad (in range of scores),
    # "middle ground", where most of the mistakes live.  In particular,
    # this scheme seems immune to all forms of "cancellation disease":  if
    # there are many strong ham *and* spam clues, this reliably scores
    # close to 0.5.  Most other schemes are extremely certain then -- and
    # often wrong.
    def spamprob(self, tokens, evidence=False):
        """
        Return best-guess probability that tokens is spam.

        tokens is an iterable object producing tokens.
        The return value is a float in [0.0, 1.0].

        If optional arg evidence is True, the return value is a pair
            probability, evidence
        where evidence is a list of (token, probability) pairs.
        """
        # We compute two chi-squared statistics, one for ham and one for
        # spam.  The sum-of-the-logs business is more sensitive to probs
        # near 0 than to probs near 1, so the spam measure uses 1-p (so
        # that high-spamprob tokens have greatest effect), and the ham
        # measure uses p directly (so that lo-spamprob tokens have greatest
        # effect).
        #
        # For optimization, sum-of-logs == log-of-product, and f.p.
        # multiplication is a lot cheaper than calling ln().  It's easy
        # to underflow to 0.0, though, so we simulate unbounded dynamic
        # range via frexp.  The real product H = this H * 2**Hexp, and
        # likewise the real product S = this S * 2**Sexp.
        H = S = 1.0
        Hexp = Sexp = 0

        clues = self._getclues(tokens)
        for prob, token in clues:
            S *= 1.0 - prob
            H *= prob
            if S < 1e-200:  # prevent underflow
                S, e = math.frexp(S)
                Sexp += e
            if H < 1e-200:  # prevent underflow
                H, e = math.frexp(H)
                Hexp += e

        # Compute the natural log of the product = sum of the logs:
        # ln(x * 2**i) = ln(x) + i * ln(2).
        S = math.log(S) + Sexp * LN2
        H = math.log(H) + Hexp * LN2

        n = len(clues)
        if n:
            S = 1.0 - chi2Q(-2.0 * S, 2 * n)
            H = 1.0 - chi2Q(-2.0 * H, 2 * n)

            # How to combine these into a single spam score?  We originally
            # used (S-H)/(S+H) scaled into [0., 1.], which equals S/(S+H).  A
            # systematic problem is that we could end up being near-certain a
            # thing was (for example) spam, even if S was small, provided that
            # H was much smaller.
            #
            # Rob Hooft stared at these problems and invented the measure we
            # use now, the simpler S-H, scaled into [0., 1.].
            prob = (S - H + 1.0) / 2.0
        else:
            prob = 0.5

        if evidence:
            clues = [(w, p) for p, w in clues]
            clues.sort(key=lambda a: a[1])
            clues.insert(0, ("*S*", S))
            clues.insert(0, ("*H*", H))
            return prob, clues
        else:
            return prob

    def add_spam(self, tokens):
        """Teach the classifier by example.

        tokens is a token stream representing a message.  If is_spam is
        True, you're telling the classifier this message is definitely spam,
        else that it's definitely not spam.
        """
        # NOTE: Graham's scheme had a strange asymmetry: when a token appeared
        # n>1 times in a single message, training added n to the token's
        # hamcount or spamcount, but predicting scored tokens only once. Tests
        # showed that adding only 1 in training, or scoring more than once when
        # predicting, hurt under the Graham scheme.
        #
        # This isn't so under Robinson's scheme, though: results improve if
        # training also counts a token only once. The mean ham score decreases
        # significantly and consistently, ham score variance decreases
        # likewise, mean spam score decreases (but less than mean ham score,
        # so the spread increases), and spam score variance increases.
        #
        # I (Tim) speculate that adding n times under the Graham scheme helped
        # because it acted against the various ham biases, giving frequently
        # repeated spam tokens (like "Viagra") a quick ramp-up in spamprob;
        # else, adding only once in training, a token like that was simply
        # ignored until it appeared in 5 distinct training spams. Without the
        # ham-favoring biases, though, and never ignoring tokens, counting n
        # times introduces a subtle and unhelpful bias.
        #
        # There does appear to be some useful info in how many times a token
        # appears in a msg, but distorting spamprob doesn't appear a correct
        # way to exploit it.
        self.store.add_spam(set(tokens))

    def add_ham(self, tokens):
        self.store.add_ham(set(tokens))

    def remove_spam(self, tokens):
        self.store.remove_spam(set(tokens))

    def remove_ham(self, tokens):
        self.store.remove_ham(set(tokens))

    # Return list of (prob, token, record) triples, sorted by increasing prob.
    # "token" is a token from tokens; "prob" is its spamprob (a float in 0.0
    # through 1.0); and "record" is token's associated WordInfo record if token
    # is in the training database, or None if it's not. No more than
    # max_discriminators items are returned, and have the strongest (farthest
    # from 0.5) spamprobs of all tokens in tokens. Tokens with spamprobs less
    # than minimum_prob_strength away from 0.5 aren't returned.
    def _getclues(self, tokens):
        tokens = set(tokens)
        counts = self.store.get_token_counts(tokens)

        clues = []
        for token in tokens:
            tup = self._worddistanceget(counts, token)
            if tup[0] >= self.minimum_prob_strength:
                clues.append(tup)
        clues.sort()

        return [
            (prob, token)
            for distance, prob, token in clues[:self.max_discriminators]
        ]

    def _worddistanceget(self, counts, token):
        tup = counts.get(token)
        if tup:
            prob = self.probability(*tup)
        else:
            prob = self.unknown_token_prob
        return abs(prob - 0.5), prob, token

    def probability(self, spamcount, hamcount):
        """Compute, store, and return prob(msg is spam | msg contains token).

        This is the Graham calculation, but stripped of biases, and stripped of
        clamping into 0.01 thru 0.99.  The Bayesian adjustment following keeps
        them in a sane range, and one that naturally grows the more evidence
        there is to back up a probability.
        """
        nspam, nham = self.store.get_nspam_nham()
        nspam = nspam or 1.0
        nham = nham or 1.0

        assert hamcount <= nham, "Token seen in more ham than ham trained."
        assert spamcount <= nspam, "Token seen in more spam than spam trained."

        hamratio = hamcount / nham
        spamratio = spamcount / nspam

        prob = spamratio / (hamratio + spamratio)

        S = self.unknown_token_strength
        StimesX = S * self.unknown_token_prob

        # Now do Robinson's Bayesian adjustment.
        #
        #         s*x + n*p(w)
        # f(w) = --------------
        #           s + n
        #
        # I find this easier to reason about like so (equivalent when
        # s != 0):
        #
        #        x - p
        #  p +  -------
        #       1 + n/s
        #
        # IOW, it moves p a fraction of the distance from p to x, and
        # less so the larger n is, or the smaller s is.
        n = hamcount + spamcount
        return (StimesX + n * prob) / (S + n)
