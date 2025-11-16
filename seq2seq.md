- how to handle different length of input and target sequences
-- add padding to the short one

1-Word 83 length dataset 3 layered Model
Total Avg Score 0.5410883693517989
Overfitted Immensely and Memorization problems


2-Word 1123 length dataset 3 layered Model
Total Avg Score 0.3939922623156501
No overfitting, generalization problems and semantic learning mistakes(!)
The model has learned that "Gerçekten mi?" is a synonym for "Seriously?" meaning learned the semantic behind the words.
Input: Gerçekten mi?
Target: ['Really', '?']
Output: ['Seriously', 'am', '?']

The model has learned that "Selam" is a greeting (like "Hello"). It's also learned that people often say "OK" and that greetings can end in "!".
Input: Selam.
Target: ['Hi', '.']
Output: ['Hello', 'OK', '!']