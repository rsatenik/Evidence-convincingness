import tensorflow as tf
import transformers


def my_models(max_length: int):
    strategy = tf.distribute.MirroredStrategy()  # Create the model under a distribution strategy scope.

    with strategy.scope():
        # Encoded token ids from BERT tokenizer
        input_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="input_ids"
        )
        # Attention masks show which tokens should be attended to
        attention_masks = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="attention_masks"
        )
        # Token type ids show different sequences in the model
        token_type_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="token_type_ids"
        )
        # Loading pretrained BERT model
        bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
        # Freeze the BERT model to reuse the pretrained features without any change
        bert_model.trainable = False

        bert_output = bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        sequence_output = bert_output.last_hidden_state

        # Add trainable layers on top of frozen layers to adapt the pretrained features
        bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(sequence_output)
        # Apply hybrid pooling approach to bi_lstm sequence output
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
        concat = tf.keras.layers.concatenate([avg_pool, max_pool])
        dropout = tf.keras.layers.Dropout(0.3)(concat)
        output = tf.keras.layers.Dense(2, activation="softmax")(dropout)
        model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks, token_type_ids], outputs=output
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )
        print(f"Strategy: {strategy}")
        model.summary()
        return model, bert_model
