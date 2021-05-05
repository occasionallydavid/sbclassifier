This package contains a stripped down version of the SpamBayes classifier, with
the following changes:

- The classifier and tokenizer code has been kept. All other code has been
  removed.
- The tokenizer has been stripped down and simplified. In particular all code
  designed specifically for email parsing has been removed.
- The remaining code has been updated and made compatible with Python 3
- The ClassifierDb class has been reduced to a simple dict subclass. The custom
  pickling code has been removed, as have all database backends.


Copyright (C) 2002-2013 Python Software Foundation; All Rights Reserved

The Python Software Foundation (PSF) holds copyright on all material
in this project.  You may use it under the terms of the PSF license;
see LICENSE.txt.

SpamBayes is a tool used to segregate unwanted mail (spam) from the mail you
want (ham).  Before SpamBayes can be your spam filter of choice you need to
train it on representative samples of email you receive.  After it's been
trained, you use SpamBayes to classify new mail according to its spamminess
and hamminess qualities.

For more details, see spambayes/README.txt.
