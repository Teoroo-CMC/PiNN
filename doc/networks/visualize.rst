Visualizing a network
=====================

TODO: Make this a colab notebook.

PiNN provides a tool called PiNNboard to visualize the activation and weights of
an ANN.

The following code trains a simple PiNet and writes the weights and activations
to the training log for visualziation.

.. code:: Python

   from tensorboard_plugin_pinnboard.summary import PiNNBoardCallback
   from tensorflow.keras.callbacks import TensorBoard

   logdir = 'logs/PiNet'
   tb_cbk = TensorBoard(log_dir=logdir, write_graph=True)
   pb_cbk = PiNNBoardCallback(logdir, train_set.apply(sparse_batch(30)))

   pinet = PiNet(pp_nodes=[6], ii_nodes=[6,6], pi_nodes=[6,6], out_nodes=[3],
                 depth=2, out_pool='sum')

   pinet.compile(optimizer='Adam', loss='MAE')
   pinet.fit(train, epochs=3, validation_data=vali, callbacks=[tb_cbk, pb_cbk])
