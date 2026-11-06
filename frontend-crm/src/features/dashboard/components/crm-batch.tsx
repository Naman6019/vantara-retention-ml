import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { UploadCloud, FileText, CheckCircle2, Download, Play, RefreshCw } from 'lucide-react'

export function CrmBatch() {
  const [fileName, setFileName] = useState<string | null>(null)
  const [isScoring, setIsScoring] = useState<boolean>(false)
  const [scoredData, setScoredData] = useState<Array<{
    customer_id: number
    persona: string
    churn_prob: number
    pred_clv: number
    playbook: string
  }> | null>(null)

  const handleSimulateSample = () => {
    setFileName('sample_batch_100_accounts.csv')
    setScoredData(null)
  }

  const handleRunScoring = () => {
    setIsScoring(true)
    setTimeout(() => {
      setScoredData([
        { customer_id: 12347, persona: 'High-Value Champions (VIP)', churn_prob: 8.5, pred_clv: 8420.5, playbook: '🌟 VIP Upsell & Early Access Privilege' },
        { customer_id: 12348, persona: 'At-Risk / Lapsed High-Value', churn_prob: 74.2, pred_clv: 5120.0, playbook: '🚨 VIP Concierge Call & Executive Retention Offer' },
        { customer_id: 12349, persona: 'Loyal / Steady Core', churn_prob: 22.1, pred_clv: 2450.0, playbook: '✅ Relationship Nurturing & Standard Engagement' },
        { customer_id: 12350, persona: 'Dormant / Low-Engagement', churn_prob: 91.8, pred_clv: 310.0, playbook: '✉️ Automated Win-Back Email Sequence & Survey' },
        { customer_id: 12351, persona: 'At-Risk / Lapsed High-Value', churn_prob: 66.4, pred_clv: 4200.0, playbook: '🎁 Priority Loyalty Incentive & CS Check-in' },
      ])
      setIsScoring(false)
    }, 1200)
  }

  const handleDownloadResults = () => {
    if (!scoredData) return
    const headers = ['Customer ID', 'Persona', 'Churn Risk (%)', 'Predicted CLV (£)', 'Recommended Playbook']
    const rows = scoredData.map((r) => [r.customer_id, `"${r.persona}"`, r.churn_prob, r.pred_clv, `"${r.playbook}"`])
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n')
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement('a')
    link.setAttribute('href', encodedUri)
    link.setAttribute('download', 'scored_crm_accounts_export.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className='space-y-6'>
      <div>
        <h2 className='text-lg font-bold tracking-tight'>Ad-Hoc Batch Scoring & CRM Sync</h2>
        <p className='text-xs text-muted-foreground'>
          Upload offline customer lists to compute multi-model risk scores, project lifetime value, and download enriched CRM sync files.
        </p>
      </div>

      {/* Step Indicator */}
      <Card className='border-border/60 shadow-sm'>
        <CardContent className='p-4'>
          <div className='flex flex-wrap items-center justify-between gap-2 text-xs'>
            <div className='flex items-center gap-2 font-medium text-foreground'>
              <span className='flex size-5 items-center justify-center rounded-full bg-indigo-600 text-[10px] text-white'>1</span>
              Upload Account Records
            </div>
            <span className='text-muted-foreground'>→</span>
            <div className='flex items-center gap-2 font-medium text-foreground'>
              <span className='flex size-5 items-center justify-center rounded-full bg-indigo-600 text-[10px] text-white'>2</span>
              Feature Schema Validation
            </div>
            <span className='text-muted-foreground'>→</span>
            <div className='flex items-center gap-2 font-medium text-foreground'>
              <span className='flex size-5 items-center justify-center rounded-full bg-indigo-600 text-[10px] text-white'>3</span>
              XGBoost & Ridge Inference
            </div>
            <span className='text-muted-foreground'>→</span>
            <div className='flex items-center gap-2 font-medium text-emerald-600 dark:text-emerald-400'>
              <span className='flex size-5 items-center justify-center rounded-full bg-emerald-600 text-[10px] text-white'>4</span>
              Export Enriched CRM File
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Upload Zone */}
      <Card className='border-border/60 shadow-sm'>
        <CardHeader>
          <CardTitle className='text-sm font-semibold'>Upload Customer Dataset (CSV)</CardTitle>
          <CardDescription className='text-xs'>Supported format: CSV with customer RFM and transaction behavior vectors</CardDescription>
        </CardHeader>
        <CardContent className='space-y-4'>
          <div className='flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border/80 bg-muted/20 p-8 text-center hover:bg-muted/40 transition-colors'>
            <div className='flex size-12 items-center justify-center rounded-full bg-indigo-50 text-indigo-600 dark:bg-indigo-950/50 dark:text-indigo-400 mb-3'>
              <UploadCloud className='size-6' />
            </div>
            <p className='text-sm font-semibold text-foreground'>
              {fileName ? fileName : 'Drag & drop account CSV file here'}
            </p>
            <p className='text-xs text-muted-foreground mt-1'>Supports files up to 200MB • UTF-8 CSV</p>

            <div className='mt-4 flex items-center gap-3'>
              <Button onClick={handleSimulateSample} variant='outline' size='sm' className='text-xs'>
                <FileText className='size-3.5 mr-1.5' />
                Load Sample 100-Account Batch
              </Button>
              {fileName && (
                <Button
                  onClick={handleRunScoring}
                  disabled={isScoring}
                  size='sm'
                  className='bg-indigo-600 hover:bg-indigo-700 text-white text-xs'
                >
                  {isScoring ? (
                    <>
                      <RefreshCw className='size-3.5 mr-1.5 animate-spin' />
                      Scoring Accounts...
                    </>
                  ) : (
                    <>
                      <Play className='size-3.5 mr-1.5' />
                      Execute Multi-Model Scoring
                    </>
                  )}
                </Button>
              )}
            </div>
          </div>

          {/* Results preview */}
          {scoredData && (
            <div className='space-y-3 pt-2'>
              <div className='flex items-center justify-between'>
                <div className='flex items-center gap-2'>
                  <CheckCircle2 className='size-4 text-emerald-600' />
                  <span className='text-xs font-semibold text-foreground'>
                    Inference Complete: 100 Accounts Scored with XGBoost & Ridge
                  </span>
                </div>
                <Button onClick={handleDownloadResults} size='sm' className='bg-emerald-600 hover:bg-emerald-700 text-white text-xs gap-1.5'>
                  <Download className='size-3.5' />
                  Download Scored CRM CSV
                </Button>
              </div>

              <div className='overflow-x-auto rounded-lg border border-border/60'>
                <table className='w-full text-left text-xs'>
                  <thead className='border-b border-border/60 bg-muted/40 font-semibold text-muted-foreground'>
                    <tr>
                      <th className='p-2.5'>Customer ID</th>
                      <th className='p-2.5'>Assigned Persona</th>
                      <th className='p-2.5'>Churn Risk</th>
                      <th className='p-2.5'>Predicted 90d CLV</th>
                      <th className='p-2.5'>Action Playbook</th>
                    </tr>
                  </thead>
                  <tbody className='divide-y divide-border/40'>
                    {scoredData.map((item) => (
                      <tr key={item.customer_id} className='hover:bg-muted/30'>
                        <td className='p-2.5 font-mono font-medium text-foreground'>#{item.customer_id}</td>
                        <td className='p-2.5'>{item.persona}</td>
                        <td className='p-2.5'>
                          <span
                            className={`rounded-md px-2 py-0.5 text-[11px] font-bold font-mono ${
                              item.churn_prob >= 70
                                ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300'
                                : item.churn_prob >= 50
                                ? 'bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300'
                                : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300'
                            }`}
                          >
                            {item.churn_prob}%
                          </span>
                        </td>
                        <td className='p-2.5 font-mono text-emerald-600 dark:text-emerald-400 font-medium'>
                          £{item.pred_clv.toLocaleString('en-GB', { minimumFractionDigits: 2 })}
                        </td>
                        <td className='p-2.5 font-medium text-foreground text-[11px]'>{item.playbook}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
