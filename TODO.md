# Pomysły, co trzeba zrobić

# PLAN: 
## artykuł: https://digital.liby.waikato.ac.nz/conferences/ivcnz07/papers/ivcnz07-paper51.pdf

* w oparciu o wykryte kontury wycinamy każdą kartę z obrazka (przykładowe wycięte karty dla "dwie_pary.jpg" w folderze "cards_cropped_from_img")
* z wyciętych kart -> patrzymy na lewy górny róg (może kolejne wycięcie?) -> klasyfikacja górnego znaku "rank" i dolnego znaku "suit" (karta z "10" będzie odrobinę problematyczna ze względu na min 3 kontury w najlepszym przypadku)



## Klasyfikacja znaku na karcie
* Trzeba zrobić zbiór "groundtruth" będący zbiorem zdjęć - przykładowych znaków typu (suit: As, Kier, Pik, Trefl i rank: 2,3, ..., 10, J, Q, K, A) do porównania konturu wyciągniętego z obrazka i klasyfikacji. Przykładowy dataset z groundtruth: https://github.com/wlllsllck/Poker_Detection/tree/master/Card_Imgs
* Jak porównać ze sobą dwa kontury/maski? Może wystarczy cv2.absdiff?
