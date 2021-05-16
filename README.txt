This package contains a stripped down version of the SpamBayes classifier, with
the following changes:

- The classifier and tokenizer code has been kept. All other code has been
  removed.
- The tokenizer has been stripped down and simplified. In particular all code
  designed specifically for email parsing has been removed.
- The ClassifierDb class has been reduced to a simple dict subclass. The custom
  pickling code has been removed, as have all database backends.
- The remaining code has been updated and made compatible with Python 3.
- An orthogonalsparse bigram (OSB) transformation has been added.
- Unicode handling has been improved.


What's it good for?
-------------------

I use sbclassifier to protect websites against contact form spam.

With a training set of a handful each of spam and non-spam messages it is
already useful. Once the training data set gets above about 20 messages of each
type I am happy to let it filter out the most obvious spam.


Usage
------

.. code::python

    import sbclassifier

    train_spam = [
        "We wholesale Masks for both adult and kids. Prices begin at $0.19 each."
        "Quickly And Effortlessly Remove Mold From Crevices In Your Home! ",
        "Good day, are you looking a good freight forwarder service in China? ",
        "I'm betting you'd like your website to generate more traffic and leads",
    ]
    train_ham = [
        "Hi, I tried to re set my password but keeps telling me email address unknown",
        "Is it possible to buy a print of the photograph on your homepage?",
        "Hello please can you close my account as soon as possible?",
        "Just wanted to say I use your website all the time, absolutely indispensible"
    ]

    # Container for the classifier data. Persist this as a pickle or write a
    # subclass that connects to a database.
    db = sbclassifier.ClassifierDb()

    classifier = sbclassifier.Classifier(db)
    for item in train_spam:
        classifier.learn(sbclassifier.tokenize_text(item), True)
    for item in train_ham:
        classifier.learn(sbclassifier.tokenize_text(item), False)

    unknown = (
        "I've helped hundreds of companies increase their traffic "
        "and I'd love to show you what my service can do for you."
    )
    probability, evidence = classifier.spamprob(sbclassifier.tokenize_text(unknown))
    print(probability)
    print(evidence)

The above script will print out::

    0.902
    [('*H*', 0.104), ('*S*', 0.908), ('can', 0.155), ('for', 0.845), ('service', 0.845), ('traffic', 0.845), ('and', 0.908)]

sbclassifier assigns 90% probability to this unknown message being spam.
It can also produce a sequence of (word, probability) pairs that reveals the
tokens that were important in this calculation.

More information
----------------

The `spambayes source repository <https://github.com/smontanaro/spambayes/>`_
contains a wealth of information on how and why the classifier works as it
works, as does the `SpamBayes wiki <http://entrian.com/sbwiki/>`_.


Copyright
---------

Copyright (C) 2002-2013 Python Software Foundation; All Rights Reserved

The Python Software Foundation (PSF) holds copyright on all material
in this project.  You may use it under the terms of the PSF license;
see LICENSE.txt.


