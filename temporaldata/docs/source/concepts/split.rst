
Splitting data into train/val/test
==================================

For machine learning applications, you can set train/validation/test domains using the set_train_domain(), set_valid_domain() and set_test_domain() methods:

.. code-block:: python

    # Create intervals for train/valid/test splits
    train_interval = Interval(0, 5.0)
    valid_interval = Interval(5.0, 7.0) 
    test_interval = Interval(7.0, 10.0)

    # Set the domains
    data.set_train_domain(train_interval)
    data.set_valid_domain(valid_interval) 
    data.set_test_domain(test_interval)


The _check_for_data_leakage() method is deprecated. Data leakage should be handled by the sampler instead of using this method.
