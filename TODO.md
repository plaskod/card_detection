# Pomysły, co trzeba zrobić

* Więcej zdjęć to zawsze lepiej niż mniej zdjęć - Paulo Coelho
* Zdjęcie kart na białym tle (trudne pod thresholding pomiędzy kartą, a tłem)
* Zdjęcia kart pokrywających się nawzajem (trudne pod wycięcie kart z obrazka, "rank" i "suit" mogą być niewidoczne i psuć kolejny krok w wycinaniu rogu karty)

## artykuł 1: https://digital.liby.waikato.ac.nz/conferences/ivcnz07/papers/ivcnz07-paper51.pdf
## artykuł 2: https://web.fe.up.pt/~niadr/PUBLICATIONS/LIACC_publications_2011_12/pdf/C62_Poker_Vision_Playing_PM_LPR_LFT.pdf


# PLAN: 
* w oparciu o wykryte kontury wycinamy każdą kartę z obrazka (przykładowe wycięte karty dla "dwie_pary.jpg" w folderze "cards_cropped_from_img")
* z wyciętych kart -> patrzymy na lewy górny/ prawy dolny róg (może kolejne wycięcie?) -> konturowanie rogu karty (wyciąganie znaku "rank" i "suit) -> **klasyfikacja** górnego znaku "rank" i dolnego znaku "suit" (karta z "10" będzie odrobinę problematyczna ze względu na min 3 kontury w najlepszym przypadku)


## Klasyfikacja znaku na karcie jak w artykule 1:
* Trzeba zrobić zbiór "groundtruth" będący zbiorem zdjęć - przykładowych znaków typu (suit: As, Kier, Pik, Trefl i rank: 2,3, ..., 10, J, Q, K, A) do porównania konturu wyciągniętego z obrazka i klasyfikacji. Przykładowy dataset z groundtruth: https://github.com/wlllsllck/Poker_Detection/tree/master/Card_Imgs
* Jak porównać ze sobą dwa kontury/maski? Może wystarczy cv2.absdiff?
