EXAMPLE GAME:

python run.py
2025-02-20 00:38:21.054924: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.   
2025-02-20 00:38:22.269616: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.   
| | | |
| | | |
| | | |
Turn: X
Player X took position (2, 2).
| | | |
| | | |
| | |X|
Turn: O
reference:
row 0 is neutral.
row 1 is happy.
row 2 is surprise.
2025-02-20 00:38:34.089073: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Emotion detected as happy (row 1). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.
text
0
reference:
col 0 is neutral.
col 1 is happy.
col 2 is surprise.
Emotion detected as neutral (col 0). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.

Player O took position (0, 0).
|O| | |
| | | |
| | |X|
Turn: X
Player X took position (1, 2).
|O| | |
| | |X|
| | |X|
Turn: O
reference:
row 0 is neutral.
row 1 is happy.
row 2 is surprise.
Emotion detected as neutral (row 0). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.

reference:
col 0 is neutral.
col 1 is happy.
col 2 is surprise.
Emotion detected as surprise (col 2). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.

Player O took position (0, 2).
|O| |O|
| | |X|
| | |X|
Turn: X
Player X took position (0, 1).
|O|X|O|
| | |X|
| | |X|
Turn: O
reference:
row 0 is neutral.
row 1 is happy.
row 2 is surprise.
Emotion detected as surprise (row 2). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.

reference:
col 0 is neutral.
col 1 is happy.
col 2 is surprise.
Emotion detected as neutral (col 0). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.

Player O took position (2, 0).
|O|X|O|
| | |X|
|O| |X|
Turn: X
Player X took position (2, 1).
|O|X|O|
| | |X|
|O|X|X|
Turn: O
reference:
row 0 is neutral.
row 1 is happy.
row 2 is surprise.
Emotion detected as neutral (row 0). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.
text
1
reference:
col 0 is neutral.
col 1 is happy.
col 2 is surprise.
Emotion detected as neutral (col 0). Enter 'text' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.
text
1
Player O took position (1, 1).
|O|X|O|
| |O|X|
|O|X|X|
Player O has won!


How well did your interface work?
    The inteface at first worked very badly to my knowledge, but after I took a look at the training images I realized that I was very far away. Once I got a lot closer and exagerated my emotions it starts working better, I also was able to find the quirks the model had where it was better at surprised faces from very close and better at neutral from medium distence. I used to only get Happy and now I can almost never get it again, honestly there is a lot of room for improvement.
Did it recognize your facial expressions with the same accuracy as it achieved against the test set?
    I would say that it was much worse at recognizing my facial expressions then in the test set, but after I figured out that i was just too far away most of the time it started performing around 75% of the time give or take.
If not, why not?
    I think that at first I was thinking that it was not performing as well because the model overfit to the data, which in a way it did because it made it reliant on distance to the camera, but if I did try to look like the training images it worked a lot better

Code :

def _get_emotion(self, img) -> int:
    resized_img = cv2.resize(img, image_size)
    
    # Convert grayscale to RGB
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    
    # Add batch dimension and normalize pixel values
    input_img = np.expand_dims(rgb_img, axis=0)

    # Load trained model if not already loaded
    if not hasattr(self, 'model'):
        self.model = load_model('results/basic_model_9_epochs_timestamp_1739866133.keras')
    
    # Make prediction
    prediction = self.model.predict(input_img, verbose=0)

    # Get the index of the highest probability class (0 for neutral, 1 for happy, 2 for surprise)
    emotion_index = int(np.argmax(prediction[0]))
    
    return emotion_index