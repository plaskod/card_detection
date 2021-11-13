# Pomysły, co trzeba zrobić


## Klasyfikacja znaku na karcie
* Trzeba zrobić zbiór "groundtruth" będący zbiorem zdjęć - przykładowych znaków typu (suit: As, Kier, Pik, Trefl i rank: 2,3, ..., 10, J, Q, K, A) do porównania konturu wyciągniętego z obrazka i klasyfikacji. Przykładowy dataset z groundtruth: https://github.com/wlllsllck/Poker_Detection/tree/master/Card_Imgs
* Jak porównać ze sobą dwa kontury/maski? Może cv2.absdiff?