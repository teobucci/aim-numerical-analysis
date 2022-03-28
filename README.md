## NumAnalysis AIM

La challenge si divide in quattro punti

- Costruire una rete Neurale che interpoli i dati restituiti da una funzione u_ex
- Costruire una PINN che interpoli i dati facendo leva su una equazione differenziale
  oltre alla pure interpolazione
- Usare la PINN per stimare il parametro mu della equazione

I parametri con cui costruire la PINN sono inseriti nel file "input/NN_params.json".

Potete modificare la struttura del file di input se cambiate come le classi vengono
definite rispetto allo scheletro dato, che consiglio di usare come punto di partenza.
L'importante è che per l'input dato l'output restituito ("output/logs.txt") sia il
piu' fedele possibile alla soluzione che vi verrà mostrata nei prossimi giorni.

Una volta che riuscite a completare l'allenamento della PINN in tutti e tre i punti,
cambiate i parametri della PINN e della ottimizzazione (usando un vostro ".json"
possibilmente), cosi' che si avvicini il piu' possibile alla soluzione reale
e il parametro mu stimato sia il piu' possibile vicino a 1.
