# Roteiro sugerido - Seminário III PitchView

Duração alvo: até 5 minutos.

## Slide 1 - Abertura
Apresentar o PitchView como uma plataforma para análise de tomada de decisão por vídeo no futebol. Explicar que o foco do Seminário III é mostrar o avanço prático: o sistema já executa visão computacional real.

## Slide 2 - Ponto de partida
Relembrar rapidamente que no Seminário II o sistema estava concentrado em upload, conversão para MP4 e planejamento do pipeline. A etapa de YOLO/tracking ainda era objetivo futuro.

## Slide 3 - Organização dos testes
Mostrar que o protótipo agora tem dashboard, histórico e status de pipeline, permitindo acompanhar quais vídeos foram enviados e quais análises foram executadas.

## Slide 4 - YOLOv8 em tempo real
Mostrar a tela Visualização da Análise. Destacar que, enquanto o vídeo é processado, o frontend recebe frames anotados do backend e atualiza progresso, jogadores detectados, bola detectada e frames processados.

## Slide 5 - Validação visual
Explicar que as caixas e labels permitem validar visualmente se o modelo está detectando jogadores e bola corretamente. Esse é o ponto central antes de avançar para tracking.

## Slide 6 - Melhoria de precisão
Explicar o problema encontrado: pessoas fora do gramado contaminavam a contagem. A solução adicionada usa OpenCV para definir a área jogável por linhas do campo e considera o ponto dos pés da detecção para aceitar ou descartar jogadores.

## Slide 7 - Resultados e próximos passos
Concluir que o TC avançou de um fluxo planejado para um protótipo funcional de visão computacional. Próximos passos: testar mais vídeos, refinar a filtragem, implementar tracking, extrair features e iniciar a camada de machine learning para análise de decisão.
