# Pomysły, co trzeba zrobić

## Jakie zdjęcia zrobić?
* "Więcej zdjęć to zawsze lepiej niż mniej zdjęć" - Paulo Coelho
* Zdjęcia kart na białym tle (trudne pod thresholding pomiędzy kartą, a tłem)
* Zdjęcia kart pokrywających się nawzajem (trudne pod wycięcie kart z obrazka, "rank" i "suit" mogą być niewidoczne i niweczyć kolejne kroki: wycinania rogu karty pod klasyfikator)
* Zdjęcia każdego możliwego znaku pod dataset z groundtruth - odnośnikiem do klasyfikacji (czyt. artykuł 1)

## artykuł 1: https://digital.liby.waikato.ac.nz/conferences/ivcnz07/papers/ivcnz07-paper51.pdf
## artykuł 2: https://web.fe.up.pt/~niadr/PUBLICATIONS/LIACC_publications_2011_12/pdf/C62_Poker_Vision_Playing_PM_LPR_LFT.pdf


# PLAN (sprytny): 
* w oparciu o wykryte kontury wycinamy każdą kartę z obrazka (przykładowe wycięte karty dla "dwie_pary.jpg" w folderze "cards_cropped_from_img")
* z wyciętych kart -> patrzymy na lewy górny/ prawy dolny róg (może kolejne wycięcie?) -> konturowanie rogu karty (wyciąganie znaku "rank" i "suit) -> **klasyfikacja** górnego znaku "rank" i dolnego znaku "suit" (karta z "10" będzie odrobinę problematyczna ze względu na min 3 kontury w najlepszym przypadku)


## Klasyfikacja znaku na karcie jak w artykule 1:
* Trzeba zrobić zbiór "groundtruth" będący zbiorem zdjęć - przykładowych znaków typu (suit: As, Kier, Pik, Trefl i rank: 2,3, ..., 10, J, Q, K, A) do porównania konturu wyciągniętego z obrazka i klasyfikacji. Przykładowy dataset z groundtruth: https://github.com/wlllsllck/Poker_Detection/tree/master/Card_Imgs
* Jak porównać ze sobą dwa kontury/maski? Może wystarczy cv2.absdiff?

## Co wymaga posprzątania na śmietnisku kodu zwanym main_Dawida.py?
**Wy śmiałkowie, którzy odważycie się zapuścić w main_Dawida.py pamiętajcie, aby sugerować się znacznikami "#TODO" i "#FIXME", a być może uda Wam się przeżyć w kontakcie z BUG-ami i Runtime error-ami**  
* przez cv.waitKey(0) i cv.destroyAllWindows() wyświetlane jest tylko jedno zdjęcie, pomimo prób przetworzenia wielu zdjęć naraz z listy selected_images za pomocą process_selected_images, najprawdopodobniej trzeba będzie zapisywać przetworzone obrazy jako matplotlib figure (jak z labów z samolotami)
* może się przyda klasa Card dziedzicząca po klasie Image, ale ta potrzeba wyjdzie później

