-- BigQuery view stroke_monitoring.prediction_daily, over the table filled by the `stroke-predictions` log sink.
SELECT
  DATE(timestamp)                    AS day,
  resource.labels.revision_name      AS revision,
  COUNT(*)                           AS predictions,
  COUNTIF(jsonPayload.prediction = 'Stroke') AS stroke_predictions,
  AVG(jsonPayload.confidence)        AS avg_confidence,
  AVG(jsonPayload.latency_ms)        AS avg_latency_ms,
  MAX(jsonPayload.latency_ms)        AS max_latency_ms
FROM `stoke-serving.stroke_monitoring.run_googleapis_com_stdout`
GROUP BY day, revision
ORDER BY day DESC
