# 📚 Miniguia de Estudos: Fundamentos e Prática de Machine Learning com Python

## 🎯 Contexto e Objetivos
Este repositório contém o meu "Caderno Temático" focado nos fundamentos de Aprendizado de Máquina (Machine Learning), unindo a teoria por trás dos algoritmos à prática com Python, Pandas e Scikit-Learn. O objetivo deste material é servir como um *brain ou external memory* para revisões rápidas e consulta de pipelines de dados.

**Objetivos de Estudo:**
1. Compreender a diferença fundamental entre aprendizado supervisionado e não supervisionado.
2. Dominar o fluxo de trabalho básico de modelagem de dados: pré-processamento, divisão de treino/teste, ajuste do modelo e validação.
3. Desmistificar o funcionamento de algoritmos baseados em árvores (Decision Trees e Random Forests).
4. Entender métricas de avaliação além da "Acurácia" (Matriz de Confusão, Precisão e Recall).
5. Utilizar o NotebookLM do Google como ferramenta de curadoria, extração de insights e síntese de documentações técnicas extensas.

---

## 📂 Curadoria de Fontes
Para alimentar a Inteligência Artificial e garantir respostas com alto rigor técnico e embasamento científico, consolidei as seguintes fontes (em formatos PDF/Web):

1. **[Documentação Oficial do Scikit-Learn](https://scikit-learn.org/stable/getting_started.html)** - Foco nos guias de usuário para classificação e métricas.
2. **[Documentação do Pandas](https://pandas.pydata.org/docs/)** - Foco em manipulação e limpeza de DataFrames.
3. **[Python Data Science Handbook (Jake VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/)** - Capítulos essenciais de introdução ao ML e feature engineering.
4. **[Glossário de Machine Learning do Google](https://developers.google.com/machine-learning/glossary)** - Foco na padronização de termos e jargões da área.

---

## 🛠️ Engenharia de Prompts e "Cicatrizes" (Troubleshooting)

Durante o processo de estudo com o NotebookLM, a forma de estruturar a pergunta alterou drasticamente a utilidade da resposta. Abaixo, documento meu processo analítico:

### Teste 1: Diferenciando os tipos de aprendizado
* **Prompt Inicial:** "Explique o que é aprendizado supervisionado e não supervisionado."
* **Resultado:** A IA gerou um texto muito acadêmico, focado em equações e sem exemplos palpáveis.
* **Cicatriz / Refinamento:** Aprendi que preciso ancorar o modelo em casos de uso de negócios para facilitar a fixação.
* **Prompt Refinado:** "Com base nos documentos, explique a diferença entre aprendizado supervisionado e não supervisionado como se você estivesse ensinando um programador júnior. Dê exatamente um exemplo prático focado em regras de negócio de uma empresa para cada um."
* **Resultado Obtido:** A IA trouxe o supervisionado como "prever a rotatividade (churn) de clientes com base no histórico de uso" e o não supervisionado como "segmentar a base de usuários por comportamento de navegação".

### Teste 2: O perigo da Acurácia
* **Prompt Inicial:** "Como saber se meu modelo de classificação está bom?"
* **Resultado:** Resposta genérica focada apenas na acurácia (porcentagem de acertos totais).
* **Cicatriz / Refinamento:** A acurácia pode ser enganosa em dados desbalanceados. Precisei forçar a IA a buscar métricas mais robustas.
* **Prompt Refinado:** "Usando o glossário do Google e a documentação do Scikit-Learn, explique por que a 'Acurácia' pode ser uma métrica perigosa em datasets desbalanceados. Introduza os conceitos de Precision e Recall com um exemplo crítico."
* **Resultado Obtido:** Excelente. A IA explicou que prever 99% de acerto em fraudes não adianta se os 1% que o modelo erra representam todas as fraudes reais. Ela introduziu a Matriz de Confusão de forma clara.

---

## 📖 Miniguia de Estudo (Entrega Final)

### 📝 Resumos Estruturados
* **Pipeline de Dados (O Ciclo de Vida):** A maior parte do trabalho em ML não é treinar o modelo, mas sim preparar os dados. O fluxo envolve: 1) Coleta, 2) Limpeza (lidar com nulos via Pandas), 3) Divisão (`train_test_split`), 4) Treinamento (`.fit()`), e 5) Avaliação (`.predict()` e métricas).
* **Random Forest (Floresta Aleatória):** Um modelo *Ensemble*. Em vez de confiar em uma única "Árvore de Decisão" (que pode sofrer *overfitting*), ele cria centenas de árvores usando amostras aleatórias dos dados e faz com que elas "votem" na resposta final. Isso garante previsões mais generalistas e robustas.
* **Avaliação de Modelos Críticos:** Em cenários de alta responsabilidade (como prever o risco de readmissão de um paciente em uma clínica ou classificar transações fraudulentas), os erros não têm o mesmo peso. Prever que um paciente está bem quando ele precisa de cuidados (Falso Negativo) costuma ser muito pior do que solicitar um exame extra por precaução (Falso Positivo). Por isso, focamos no **Recall** para minimizar os Falsos Negativos.

### 💻 Snippet de Código: Treinando o Primeiro Modelo
Um exemplo rápido de como esses conceitos se traduzem em código usando a stack padrão do ecossistema Python:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Carregando e separando os dados (Features vs Label)
# Exemplo: Prever se um usuário vai assinar um plano premium (1) ou não (0)
X = dataset.drop('assinou_premium', axis=1) # Features (idade, tempo_de_tela, cliques)
y = dataset['assinou_premium']              # Label (Alvo)

# 2. Divisão de Treino e Teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Inicializando e Treinando o Modelo
modelo_rf = RandomForestClassifier(n_estimators=100)
modelo_rf.fit(X_train, y_train)

# 4. Fazendo previsões e Avaliando
previsoes = modelo_rf.predict(X_test)
print(classification_report(y_test, previsoes))
```

📚 Glossário de Sobrevivência
Features (Recursos): As variáveis independentes do seu modelo (as colunas do seu DataFrame que contêm as informações).

Label (Rótulo/Alvo): A variável dependente; aquilo que você está tentando prever.

Overfitting (Sobreajuste): Quando o modelo "decora" os dados de treino, acertando tudo neles, mas tem um desempenho péssimo em dados novos.

Matriz de Confusão: Uma tabela que mostra o desempenho do modelo de classificação, dividindo os resultados em Verdadeiros Positivos, Verdadeiros Negativos, Falsos Positivos e Falsos Negativos.

Recall (Revocação): De todos os casos que eram realmente positivos, quantos o modelo conseguiu identificar corretamente?

Precision (Precisão): De todas as vezes que o modelo disse que era positivo, quantas ele realmente acertou?

🤖 Prompts Reutilizáveis (Para revisões e novos estudos)
"Analise o documento e crie uma tabela comparativa listando 3 vantagens e 3 desvantagens do algoritmo [INSERIR ALGORITMO]. Em que tipo de dataset ele teria o pior desempenho possível?"

"Aja como um Engenheiro de Machine Learning Sênior fazendo code review. Eu escrevi o seguinte pipeline em Python: [INSERIR CÓDIGO]. Aponte falhas de segurança de dados ou vazamento de dados (data leakage) e sugira melhorias."

"Estou confuso sobre a diferença entre [MÉTRICA A] e [MÉTRICA B]. Crie uma analogia simples do dia a dia, sem usar jargões matemáticos, para eu entender quando devo priorizar uma em vez da outra."

Projeto desenvolvido com foco em Aprendizagem Ativa e Inteligência Artificial para o Desafio de Projeto da DIO.
