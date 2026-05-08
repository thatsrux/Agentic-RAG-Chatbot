---
source: https://corsi.unisa.it/uploads/rescue/499/3112/tracce20-21.pdf
found_in: https://corsi.unisa.it/information-Engineering-for-digital-medicine/immatricolazioni
type: pdf
---

Prova di verifica della preparazione personale per l’iscrizione alla Laurea 
Magistrale del 24/02/2020 
 
Esercizio n.1 
Scrivere in un linguaggio di programmazione a scelta una funzione (o un metodo di una 
classe) che riceva come parametri di input due array di numeri interi della stessa lunghezza 
(la lunghezza è essa stessa un parametro di input della funzione), e che restituisca come 
valore di ritorno un valore logico che sia vero se il primo array contiene gli stessi elementi del 
secondo ma in ordine invertito, falso altrimenti. In altre parole, la funzione restituisce un 
valore logico vero se e solo se il primo elemento del primo array è uguale all’ultimo del 
secondo array, e il secondo elemento del primo array è uguale al penultimo del secondo array 
e così via.  
Esempio: Se il primo array è [10 20 42 7 40] e il secondo è [40 7 42 20 10], la funzione 
restituisce true. Altro esempio: Se il primo array è [10 20 42 7 33 90] e il secondo è [90 33 42 
7 20 10]. la funzione restituisce false. 
 
Esercizio n.2 
Eseguire la progettazione concettuale e logica di una base dati relazionale per gestire le 
seguenti informazioni relative alla gestione di progetti software: 
• Developer, dove ogni developer (sviluppatore) ha un nome e un codice identificativo. 
• WorkPackage, dove un work package (unità di lavoro) ha un codice identificativo, una 
descrizione, un responsabile (che è un developer), un insieme di partecipanti al work 
package (che sono developer), e un insieme di deliverable prodotti dal work package. 
Si noti che un developer può essere responsabile di più work package, e può 
partecipare a più work package. 
• Deliverable, dove un deliverable (risultato) ha un codice identificativo e una 
descrizione. Si noti che un deliverable appartiene a un solo work package, mentre un 
work package può avere più deliverable. 
 
Esercizio n.3 
Realizzare mediante porte logiche elementari (AND, OR, NOT) una rete combinatoria che 
riceva in ingresso un numero naturale n rappresentato su 2 bit (il numero ha valori tra 0 e 3), 
e produca in uscita il numero naturale n+1 rappresentato su 3 bit (il numero ha valori tra 1 e 
4). La progettazione deve includere la minimizzazione della rete combinatoria. 
 
Esercizio n.4 
Calcolare il limite per n → della seguente funzione: 
  n  [exp(x + 1/n) − exp(x)] 
 
 
Esercizio n.5 
 
 


Nel sistema illustrato nella figura soprastante, la pallina nel punto A si muove verso destra 
con velocità iniziale v0= 2 m/s. Supponendo che la distanza tra il punto B e il punto C in cui la 
pallina tocca il suolo sia di 4 m, calcolare l’altezza iniziale della pallina (distanza tra A e B), 
supponendo trascurabili gli attriti e la resistenza dell’aria. 
 
 
Esercizio n.6 
Nel circuito descritto nella figura sottostante, determinare il valore della tensione V1, sapendo 
che Is è un generatore di corrente ideale la cui corrente è 100 mA e i tre resistori hanno una 
resistenza R=100 Ω. 
 
 
 
 
 
Esercizio n.7 
Un cassetto contiene 3 calzini rossi e 3 calzini blu. Estraendo al buio 3 calzini dal cassetto 
(senza rimettere dentro i calzini estratti), qual è la probabilità che almeno uno di essi sia blu? 
 
 
 
Esercizio n.8 
Scrivere la funzione di trasferimento (nel dominio di Laplace) del sistema illustrato nella 
seguente figura: 
 
 
 
 


Prova di verifica della preparazione personale per l’ammissione alla Laurea Magistrale in Ingegneria Informatica e alla Laurea Magistrale in Digital Health and Bioinformatic Engineering 28 settembre 2020  ESERCIZIO 1 Scrivere una funzione in C o un metodo in Java che, dati come parametri di ingresso due array a di n elementi e b di m elementi, restituisca un valore logico vero se e solo se tutti gli elementi di a sono minori di tutti gli elementi di b (quindi la funzione deve restituire un valore falso se almeno un elemento di a è maggiore o uguale di uno degli elementi di b).  ESERCIZIO 2 Dato un albero binario di ricerca rappresentato con la seguente struttura dati: typedef struct TNode {  int info;  struct TNode *left;  struct TNode *right; } TNode; typedef TNode *TBinaryTree; realizzare una funzione in C che prenda come parametro di ingresso un albero e restituisca un valore logico vero se e solo se tutti gli elementi dell’albero sono maggiori di 0.  ESERCIZIO 3 Progettare, minimizzare e implementare in termini di porte logiche elementari la rete combinatoria corrispondente alla funzione di quattro variabili f(a,b,c,d) che vale 1 se e solo se a ha lo stesso valore di almeno una delle altre tre variabili (b, c oppure d).  ESERCIZIO 4 Calcolare il punto di minimo (verificando l’esistenza di un minimo) della funzione:    𝑥(𝑡)=𝑒("#$%)−2∙𝑡∙𝑒"  ESERCIZIO 5 Dato il pendolo semplice illustrato in figura: 
 supponendo che all’istante t0, in cui q0 = π/3 rad, la velocità della massa sia v0 = 0.5 m/s, calcolare la velocità v nell’istante in cui la massa m passa per il punto di equilibrio (q = 0 rad). Si assuma trascurabile l’attrito dell’aria e la massa del filo, e che la massa m sia puntiforme.  


 ESERCIZIO 6 Dato il seguente circuito: 
  con V1=1.5 V, V2=3 V, R1=R2=R3=100 W, calcolare la potenza dissipata dal resistore R3.  ESERCIZIO 7 Data la risposta a gradino unitario rappresentata nella seguente figura: 
 determinare a quale tra i seguenti sistemi si può riferire: 𝐺!(𝑠)=20𝑠∙(𝑠+1)						𝐺"(𝑠)=20𝑠(𝑠+1)(𝑠+10)							𝐺#(𝑠)=10𝑠(𝑠+5)(𝑠+10)					𝐺$(𝑠)=40(𝑠+1)(𝑠+10) La risposta deve essere motivata, e si deve anche spiegare perché si sono esclusi gli altri sistemi.  ESERCIZIO 8 Supponendo di avere tre monete non truccate, ciascuna con una faccia “Testa” e una faccia “Croce”, e di lanciare insieme le tre monete per due volte, qual è la probabilità che nei due lanci esca lo stesso numero di Teste? 


Prova di verifica della preparazione personale per l’ammissione alla Laurea Magistrale in Ingegneria Informatica e alla Laurea Magistrale in Digital Health and Bioinformatic Engineering 4 dicembre 2020  ESERCIZIO 1 Scrivere una funzione in C o un metodo in Java che, dati come parametri un array di interi a di n elementi, e un intero x, rimuova dall’array a tutti gli elementi maggiori di x. La funzione deve restituire come valore di ritorno il nuovo numero di elementi dell’array a. NOTA BENE: la funzione deve rimuovere gli elementi dall’array, non impostarli a 0 o a un altro valore; al termine della funzione, l’array deve contenere solo quegli elementi che non erano maggiori di x.  ESERCIZIO 2 Data una lista concatenata rappresentata con la seguente struttura dati: typedef struct TNode {  int info;  struct TNode *next; } TNode; typedef TNode *TList; realizzare una funzione ricorsiva in C che prenda come parametro di ingresso una lista e restituisca un valore logico vero se e solo se tutti gli elementi della lista sono in ordine crescente. NOTA BENE: la funzione deve essere ricorsiva. Calcolare la complessità computazionale della funzione usando le formule di ricorrenza.  ESERCIZIO 3 Scrivere in Java una classe Contatore, che consenta di leggere e di incrementare un contatore intero, inizializzato a 0, in maniera thread-safe; i metodi pubblici della classe devono essere incrementa, che aggiunge 1 al contatore, e getValore, che restituisce il valore corrente del contatore. Scrivere un main che crea un’istanza della classe Contatore e quindi crea e fa partire 10 threads, ciascuno dei quali incrementa il contatore e stampa a video il valore del contatore dopo l’incremento.  ESERCIZIO 4 Dato il seguente sistema di equazioni:     !𝑥+𝑎∙𝑦	=	32𝑥−𝑦	=	5−2∙𝑎  in cui x e y sono le incognite, e a è un parametro costante, indicare per quali valori di a: • il sistema ammette una soluzione unica; • il sistema non ammette soluzioni; • il sistema ammette infinite soluzioni.  ESERCIZIO 5       
v0 
A 
B 

 Nella figura precedente, la pallina viene lanciata dal punto A di coordinate (0 m, h) con velocità orizzontale v0=5 m/s, e atterra nel punto B di coordinate (10 m, 0 m). Determinare l’altezza h del punto A, considerando trascurabili gli attriti e la resistenza dell’aria.  ESERCIZIO 6 Dato il seguente circuito: 
 con R1=R2=R3=300 W, sapendo che la potenza erogata dal generatore di tensione è P=4 W, calcolare la tensione V del generatore di tensione.  ESERCIZIO 7 Dato il diagramma di Bode nella seguente figura: 
  determinare quale, tra le seguenti, è la funzione di trasferimento del sistema rappresentato, indicando esplicitamente per ciascuna delle altre funzioni i motivi per cui possono essere escluse: 𝐺!(𝑠)=−20∙𝑠(𝑠−20)"						𝐺"(𝑠)=−200∙𝑠(𝑠"+4𝑠+20)							𝐺#(𝑠)=−200∙𝑠(𝑠+20)"					𝐺$(𝑠)=200𝑠∙(𝑠+20)"     


ESERCIZIO 8 Il segnale 𝑥(𝑡)=𝑐𝑜𝑠3!"𝑡4+𝑐𝑜𝑠3#!$𝑡4 viene campionato idealmente con una frequenza 𝑓%. a) Nel caso 𝑓%=2	Hz, indicare se è possibile ricostruire perfettamente il segnale tramite un filtro passa-basso ideale, e in caso affermativo specificare una possibile frequenza di taglio per il filtro in questione.  b) Nel caso 𝑓%=1	Hz, indicare se è possibile ricostruire perfettamente il segnale tramite un filtro passa-basso ideale, e in caso affermativo specificare una possibile frequenza di taglio per il filtro in questione.  

Prova di verifica della preparazione personale per l’ammissione alla Laurea Magistrale in Ingegneria Informatica e alla Laurea Magistrale in Digital Health and Bioinformatic Engineering 5 febbraio 2021  ESERCIZIO 1 Scrivere, in un linguaggio di programmazione a scelta dello studente, una funzione o un metodo che, dati come parametri di ingresso due array di interi, l’array a di n elementi e l’array b di m elementi, restituisca attraverso un parametro di uscita l’array c formato dai soli elementi che sono presenti sia in a che in b indipendentemente dalla posizione. La funzione deve restituire come valore di ritorno il numero di elementi dell’array c. NOTA BENE: gli elementi da restituire nell’array c (che sono presenti sia in a che in b) potrebbero trovarsi in posizioni diverse nei due array iniziali.  Esempio: Se a={10, 7, 14, 9} e  b={9, 25, 42, 38, 14}, allora in uscita l’array c dovrà contenere  {14, 9} e il valore di ritorno della funzione dovrà essere 2. Nota: indicare nella soluzione il linguaggio di programmazione utilizzato.  ESERCIZIO 2 In un linguaggio di programmazione a scelta dello studente, definire la struttura dati per un albero binario di ricerca, i cui nodi contengano valori interi, e realizzare una funzione o un metodo che prenda come parametro di ingresso un albero, e restituisca come valore di ritorno l’informazione (intera) della foglia più piccola contenuta nell’albero. Calcolare la complessità computazionale della funzione/metodo usando le formule di ricorrenza. Nota: indicare nella soluzione il linguaggio di programmazione utilizzato.   ESERCIZIO 3 Scrivere, in un linguaggio di programmazione orientato agli oggetti a scelta dello studente, una classe Semaforo che abbia come struttura dati un contatore intero, e come metodi (oltre al costruttore): • Un metodo incrementa, che aumenta di 1 il valore del contatore • Un metodo decrementa, che, se il contatore è maggiore di 0, decrementa di 1 il contatore; se invece il contatore è 0, mette il thread corrente in attesa che il contatore diventi maggiore di 0, e poi decrementa di 1 il contatore. Nota: La classe deve essere thread-safe. Nota: indicare nella soluzione il linguaggio di programmazione utilizzato.   ESERCIZIO 4 Determinare, se ve ne sono, i punti di minimo della seguente funzione:    𝑓(𝑥)=3!"−3"#$     ESERCIZIO 5 Una automobilina radiocomandata, di massa m=500 grammi, sale sul piano inclinato illustrato nella figura sottostante (che ha un’inclinazione di π/6 rad rispetto al piano orizzontale) con velocità uniforme v = 1.5 m/s. Supponendo trascurabili le forze di attrito e di resistenza dell’aria, determinare il lavoro svolto dall’automobilina in un intervallo di tempo t = 4 secondi.  

    ESERCIZIO 6 Dato il seguente circuito: 
 in cui: R1=R2=R3=100 W, R4=300 W, e il generatore ideale di corrente genera Ig=0.3 A, determinare la potenza dissipata dal resistore R2.    ESERCIZIO 7 Dato un sistema la cui funzione di trasferimento è: 𝐺(𝑠)=1𝑠∙(𝑠+1) se il sistema riceve come ingresso 𝑢(𝑡)=𝟏(𝑡)−𝟏(𝑡−10) (vedi figura sottostante), 
  determinare quale tra le seguenti quattro è la risposta del sistema: 


A)  
B)   
C)   
D)   Indicare esplicitamente i motivi per cui può essere esclusa ciascuna delle altre risposte.    ESERCIZIO 8 Il segnale 𝑥(𝑡)=𝑠𝑖𝑛𝑐(2∙10!∙𝑡)+𝑠𝑖𝑛𝑐(10!∙𝑡)   [t espresso in secondi] Deve essere convertito in digitale. Come primo passo, il segnale deve essere campionato.  a) Calcolare la banda monolatera del segnale x(t).  b) L’operazione di campionamento è reversibile? In altri termini, sarà possibile ricostruire il segnale x(t) dai suoi campioni?  c) Se la risposta alla precedente domanda è sì, determinare la frequenza di campionamento minima che garantisce la perfetta ricostruzione del segnale x(t) e specificare (ad esempio, in forma grafica) un possibile filtro di ricostruzione ideale.  


Prova di verifica della preparazione personale per l’ammissione alla Laurea 
Magistrale in Ingegneria Informatica e alla Laurea Magistrale in Digital Health 
and Bioinformatic Engineering 
1 ottobre 2021 
 
 
ESERCIZIO 1 
Scrivere, in un linguaggio di programmazione a scelta dello studente, una funzione o un metodo che, 
dati come parametri di ingresso un array a di interi, il numero n di elementi di a, e un valore intero 
x, restituisca attraverso i suoi parametri di uscita un array a1 contenente tutti gli elementi di a minori 
di x, il numero n1 di elementi dell’array a1, un array a2 contenente tutti gli elementi di a maggiori 
o uguali di x e il numero n2 di elementi dell’array a2. La funzione non deve svolgere operazioni di 
input/output.  
Esempio: Se la funzione viene richiamata con i parametri di ingresso a={10,-4,2,0,-1, 7,3}, 
n=7 e x=2, la funzione dovrà restituire attraverso i suoi parametri di uscita a1={-4,0,-1}, n1=3, 
a2={10,2,7,3}, n2=4. 
Nota: indicare nella soluzione il linguaggio di programmazione utilizzato. 
 
 
ESERCIZIO 2 
In un linguaggio di programmazione a scelta dello studente: 
• definire la struttura dati per  una lista concatenata  semplice, i cui nodi contengano valori 
interi 
• realizzare una funzione o un metodo  che prenda come parametr i di ingresso due liste 
concatenate lst1 e lst2, e restituisca un valore logico vero se e solo se lst2 è uguale alla 
parte finale di lst1. Esempi:  
o se lst1 contiene i valori 10, 4, 7, 9  e lst2 contiene i valori 7, 9, allora la 
funzione deve restituire un valore vero  
o se lst1 contiene i valori 10, 4, 7, 9 e lst2 contiene i valori 4, 7, 9, allora 
la funzione deve restituire un valore vero  
o se lst1 contiene i valori 10, 4, 7, 9  e lst2 contiene i valori 4, 9, allora la 
funzione deve restituire un valore falso  
o se lst1 contiene i valori 10, 4, 7, 9 e lst2 contiene i valori 7, 9, 4, allora 
la funzione deve restituire un valore falso 
• Calcolare la  complessità computazionale  temporale della funzione /metodo definiti 
(indicando tutti i passaggi necessari per arrivare al risultato). 
Nota: è consentito definire funzioni ausiliarie (fornendone il codice completo) richiamate dalla 
funzione richiesta dalla traccia. 
Nota: indicare nella soluzione il linguaggio di programmazione utilizzato. 
 
 
  

ESERCIZIO 3 
Considerato un database il cui modello concettuale è  rappresentato dal seguente diagramma entità -
relazioni: 
 
a) eseguire la progettazione logica del database usando il modello relazionale 
b) verificare se il database rispetta la terza forma normale (3NF), motivando la risposta, e in caso 
negativo, normalizzarlo 
c) realizzare una query in linguaggio SQL per estrarre i nomi e i cognomi di tutti gli attori che abbiano 
partecipato ad almeno un film in cui era presente l’attore Nicholas Cage. 
 
 
ESERCIZIO 4 
Dati i seguenti vettori: 
v1=( 1,  2, 0, -1,  1) 
v2=(-1, -3, 1,  2, -1) 
v3=( 1,  0, a,  b,  c) 
determinare per quali valori dei parametri a, b e c lo spazio vettoriale generato da {v 1, v2} è uguale 
allo spazio vettoriale generato da {v1, v2, v3}. 
 
 
ESERCIZIO 5 
Un blocchetto di massa m=1 kg e di dimensioni trascurabili, scivola verso il basso lungo il piano 
inclinato rappresentato nella figura: 
 
 
 
 
P0 = (0 m, 1 m) 
P1 = (1 m, 0 m) 
 = /4 rad 
m = 1 kg 
v1 = 2 m/s 
kd = 1 
Sapendo che nel punto P 1 la velocità è pari a v1=2 m/s, e che il coefficiente di attrito dinamico è 
kd = 1, determinare la velocità iniziale del blocchetto nel punto P0.  
 

ESERCIZIO 6 
Dato il seguente circuito: 
 
in cui: R 1=R2=R3=100 , e R 4=R5=R6=50 , sapendo che la potenza dissipata dal resistore R 2 è 
P2=4 W, determinare la tensione del generatore di tensione ideale Vg. 
 
 
 
ESERCIZIO 7 
 
𝐺1 = 1
3
1
𝑠2 + 0.4𝑠 + 1 ,                       𝐺2 = 1
s2 + 2s + 1 , 
 
𝐺3 = 1
3
1
𝑠2 − 2𝑠 + 1 ,                        𝐺4 = 1
3
1
𝑠2 + 1.66𝑠 + 1 
 
Un sistema è sottoposto ad un ingresso a gradino 𝑢(𝑡) = 3 ⋅ 1(𝑡) . La risposta ottenuta 𝑦(𝑡) è 
mostrata in figura. Quale è, tra le quattro sopra elencate, la funzione di trasferimento del sistema?  
Indicare esplicitamente i motivi per cui può essere esclusa ciascuna delle altre risposte.  


ESERCIZIO 8 
Un decoder digitale elabora una stringa di 8 b it. La probabilità che un bit sia erroneamente 
decodificato è pari a 10-3, e gli eventi errore associati ai singoli bit sono indipendenti.  
 
1) Calcolare la probabilità che il secondo bit e il terzo bit siano entrambi sbagliati. 
2) Calcolare la probabilità che almeno un bit sia sbagliato. 
3) Calcolare la probabilità che il numero di bit sbagliati sia minore di 2. 
4) Cosa accade alla probabilità indicata al punto precedente se la lunghezza della stringa tende 
a infinito? 
 
 

Prova di verifica della preparazione personale per l’ammissione alla Laurea 
Magistrale in Ingegneria Informatica e alla Laurea Magistrale in Digital Health 
and Bioinformatic Engineering 
17 dicembre 2021 
 
 
ESERCIZIO 1 
Scrivere, in un linguaggio di programmazione a scelta dello studente, una funzione o un metodo che, 
dati come parametri di ingresso un array a1 di interi, il numero n1 di elementi di a1, e un array a2 
di interi, e il numero n2 di elementi di a2, fornisca al chiamante, attraverso i suoi parametri di uscita 
e/o il valore di ritorno, un array a3 contenente tutti gli elementi di a1 che siano anche presenti in a2, 
e il numero n3 di elementi dell’array a3. La funzione non deve svolgere operazioni di input/output 
e non deve modificare gli array a1 e a2.  
Esempio: Se la funzione viene richiamata con i parametri di ingresso a1={5,-4,2,0,-1,5,3}, 
n1=7 , a2={-1,5,-1,0}, n2=4, la funzione dovrà fornire al chiamante  attraverso i suoi 
parametri di uscita e/o il valore di ritorno a3={5,0,-1,5}, n3=4. 
Nota: indicare nella soluzione il linguaggio di programmazione utilizzato. 
 
 
ESERCIZIO 2 
In un linguaggio di programmazione a scelta dello studente: 
• definire la struttura dati per un albero binario di ricerca, i cui nodi contengano valori interi 
• realizzare una funzione o un metodo che prenda come parametri di ingresso un albero binario 
di ricerca tree e restituisca un valore logico true se almeno un nodo dell’albero contiene un 
valore che è esattamente il doppio del valore del suo figlio sinistro; altrimenti, se nessun nodo 
soddisfa la condizione indicata, il valore restituito deve essere false. 
• Calcolare la  complessità computazionale  temporale asintotica della funzione /metodo 
rispetto al numero di elementi dell’albero (indicando tutti i passaggi necessari per arrivare al 
risultato). 
 
Nota: indicare nella soluzione il linguaggio di programmazione utilizzato. 
 
 
ESERCIZIO 3 
Realizzare mediante porte logiche elementari (AND, OR, NOT) una rete combinatoria che riceva in 
ingresso un numero naturale n rappresentato su 2 bit (il numero ha valori tra 0 e 3), e produca in uscita 
il numero naturale n–1 rappresentato su 2 bit (se il valore di n è 0, i due bit di uscita devono avere il 
valore 112). La progettazione deve includere la minimizzazione della rete combinatoria. 
 
 
ESERCIZIO 4 
Calcolare il limite per 𝑦 → ∞ della seguente funzione: 
𝑦 ∙ (cos(2𝑥 + 1 𝑦⁄ ) − cos(2𝑥)) 
 
 
  

ESERCIZIO 5 
Nella figura sottostante, la pallina viene lanciata dal punto A con velocità iniziale v 0 e angolo 
=/4 rad. Supponendo trascurabile la resistenza dell’aria, qual è il valore minimo di v 0 necessario 
affinché la pallina atterri al di là del punto B, la cui distanza da A è 5.0 m? 
 
.  
 
 
ESERCIZIO 6 
Dato il seguente circuito: 
 
in cui: R 1=R2=R3=120 , e R 4=R5=R6, sapendo che la potenza erogata dal  generatore di tensione 
ideale Vg è di 36.0 W, e che la potenza dissipata su R1 è 4.8 W, determinare il valore di R4. 
 
 
v0 
  

ESERCIZIO 7 
 
𝐺1 = 12.5 𝑠 +  10 
𝑠2 +  11 𝑠 +  10 ,                       𝐺2 = 1
s2 + 2s + 1, 
𝐺3 = 1
𝑠2 − 2𝑠 + 1 ,                            𝐺4 = 1
𝑠2 + 1.66𝑠 + 1 
 
Un sistema è sottoposto ad un ingresso a gradino unitario 𝑢(𝑡) = 1(𝑡) . La risposta ottenuta 𝑦(𝑡) è 
mostrata in figura. Quale è, tra le quattro sopra elencate, la funzione di trasferimento del sistema? 
Giustificare la risposta. 
 
 
 
 
 
ESERCIZIO 8 
Una sorgente di informazione trasmette tre segnali: 𝑥1(𝑡) = 𝐴 cos(2𝜋𝑓0𝑡) con probabilità p1; 
𝑥2(𝑡) = 0 con probabilità p2; 𝑥3(𝑡) = −𝐴 cos(2𝜋𝑓0𝑡) con probabilità p3. 
a) Calcolare la trasformata di Fourier dei tre segnali 
b) Calcolare il valore atteso della potenza trasmessa dalla sorgente nel caso di segnali 
equiprobabili (𝑝1 = 𝑝2 = 𝑝3). 
c) Calcolare il valore atteso della potenza trasmessa dalla sorgente nel caso in cui  
𝑝1 = 𝑝3 = 1/2 ∙ 𝑝2 
 
