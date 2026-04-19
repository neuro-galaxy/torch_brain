
Splitting data into train/val/test
==================================

For machine learning applications, you can create split masks to separate a Data object into training/validation/test sets using the set_train_domain(), set_valid_domain() and set_test_domain() methods:

.. code-block:: python

    # Create intervals for train/valid/test splits
    train_interval = Interval(0, 5.0)
    valid_interval = Interval(5.0, 7.0) 
    test_interval = Interval(7.0, 10.0)

    # Set the domains and create split masks
    data.set_train_domain(train_interval)
    data.set_valid_domain(valid_interval) 
    data.set_test_domain(test_interval)


Under the hood, when setting train/valid/test domains, each data point and interval is 
labeled with its corresponding split ("train", "valid", or "test"). The _check_for_data_leakage() 
method uses these labels to verify that no data points or intervals are assigned to multiple splits, which would constitute data leakage.
