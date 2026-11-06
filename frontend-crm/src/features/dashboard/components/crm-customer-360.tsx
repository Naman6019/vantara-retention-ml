import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { PhoneCall, Gift, Mail, Sparkles, Brain, CheckCircle2 } from 'lucide-react'
import crmData from '@/data/crm-data.json'

export function CrmCustomer360() {
  const { customer_profiles } = crmData
  const availableIds = Object.keys(customer_profiles)
  const [selectedId, setSelectedId] = useState<string>(availableIds[0] || '12346')

  const currentProfile = customer_profiles[selectedId as keyof typeof customer_profiles] || customer_profiles['12346']

  const getTierBadge = (spend: number) => {
    if (spend >= 25000) return <Badge className='bg-indigo-600 text-white font-medium'>Tier 1: Enterprise VIP Champion</Badge>
    if (spend >= 5000) return <Badge className='bg-emerald-600 text-white font-medium'>Tier 2: Core Growth Account</Badge>
    return <Badge variant='outline' className='border-amber-400 bg-amber-50 text-amber-800 dark:bg-amber-950/50 dark:text-amber-300'>Tier 3: Standard Client</Badge>
  }

  const getPlaybookIcon = (playbook: string) => {
    if (playbook.includes('VIP Concierge')) return <PhoneCall className='size-4 text-rose-600' />
    if (playbook.includes('Loyalty')) return <Gift className='size-4 text-amber-600' />
    if (playbook.includes('Win-Back')) return <Mail className='size-4 text-indigo-600' />
    return <Sparkles className='size-4 text-emerald-600' />
  }

  return (
    <div className='space-y-6'>
      <div className='flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between'>
        <div>
          <h2 className='text-lg font-bold tracking-tight'>Customer 360 & Explainable AI (XAI) Dossier</h2>
          <p className='text-xs text-muted-foreground'>
            Individual account health diagnostic powered by <b>Local SHAP Feature Attribution</b> to isolate exact risk triggers.
          </p>
        </div>

        {/* Account Selector */}
        <div className='flex items-center gap-2'>
          <span className='text-xs font-medium text-muted-foreground'>Select Account:</span>
          <select
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
            className='h-8 rounded-md border border-input bg-background px-3 py-1 text-xs font-mono font-medium shadow-sm focus:outline-none focus:ring-1 focus:ring-ring'
          >
            {availableIds.map((id) => (
              <option key={id} value={id}>
                Account #{id} ({customer_profiles[id as keyof typeof customer_profiles]?.persona})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Customer 360 Header Dossier */}
      <Card className='border-border/60 bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-950 text-white shadow-md'>
        <CardContent className='p-6'>
          <div className='flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between'>
            <div className='flex items-center gap-4'>
              <div className='flex size-14 items-center justify-center rounded-2xl bg-indigo-600 text-xl font-bold text-white shadow-lg shadow-indigo-600/30'>
                {String(currentProfile.customer_id).slice(-2)}
              </div>
              <div>
                <div className='flex items-center gap-2.5'>
                  <h3 className='text-xl font-bold tracking-tight'>Account #{currentProfile.customer_id}</h3>
                  {getTierBadge(currentProfile.spend)}
                </div>
                <div className='mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-300'>
                  <span>Persona: <b className='text-white'>{currentProfile.persona}</b></span>
                  <span>•</span>
                  <span>Order Frequency: <b className='text-white'>{currentProfile.frequency} orders</b></span>
                  <span>•</span>
                  <span>Recency: <b className='text-white'>{currentProfile.recency} days ago</b></span>
                </div>
              </div>
            </div>

            <div className='rounded-xl border border-white/10 bg-white/5 p-3.5 sm:text-right backdrop-blur-sm'>
              <div className='text-[10px] font-semibold uppercase tracking-wider text-slate-400'>
                Predicted Forward 90-Day CLV
              </div>
              <div className='text-2xl font-bold text-emerald-400 font-mono'>
                £{currentProfile.pred_clv.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div className='text-[10px] text-slate-400'>Ridge ML Regression</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Churn Risk & Economics Grid */}
      <div className='grid gap-6 md:grid-cols-3'>
        {/* Churn Propensity Meter */}
        <Card className='border-border/60 shadow-sm'>
          <CardHeader className='pb-2'>
            <CardTitle className='text-sm font-semibold'>Churn Propensity Score</CardTitle>
            <CardDescription className='text-xs'>Calibrated probability of account lapse</CardDescription>
          </CardHeader>
          <CardContent className='pt-2 flex flex-col items-center justify-center text-center'>
            <div className='relative my-3 flex items-center justify-center'>
              <div
                className={`text-4xl font-extrabold font-mono ${
                  currentProfile.churn_prob >= 70
                    ? 'text-rose-600 dark:text-rose-400'
                    : currentProfile.churn_prob >= 50
                    ? 'text-amber-600 dark:text-amber-400'
                    : 'text-emerald-600 dark:text-emerald-400'
                }`}
              >
                {currentProfile.churn_prob}%
              </div>
            </div>
            <span
              className={`rounded-full px-3 py-1 text-xs font-bold ${
                currentProfile.churn_prob >= 70
                  ? 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300'
                  : currentProfile.churn_prob >= 50
                  ? 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
                  : 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
              }`}
            >
              {currentProfile.churn_prob >= 70
                ? 'Critical Churn Risk (Immediate Action)'
                : currentProfile.churn_prob >= 50
                ? 'Moderate Churn Risk (At-Risk)'
                : 'Healthy Account (Low Churn)'}
            </span>
          </CardContent>
        </Card>

        {/* Account Economics & Playbook */}
        <Card className='md:col-span-2 border-border/60 shadow-sm'>
          <CardHeader className='pb-2'>
            <CardTitle className='text-sm font-semibold'>Account Economics & Next Best Action</CardTitle>
            <CardDescription className='text-xs'>Prescriptive retention intervention for this account</CardDescription>
          </CardHeader>
          <CardContent className='space-y-4 pt-2 text-xs'>
            <div className='grid grid-cols-2 gap-4 rounded-lg border border-border/40 bg-muted/20 p-3'>
              <div>
                <div className='text-muted-foreground'>Lifetime Spend:</div>
                <div className='text-base font-bold font-mono text-foreground'>
                  £{currentProfile.spend.toLocaleString('en-GB', { minimumFractionDigits: 2 })}
                </div>
              </div>
              <div>
                <div className='text-muted-foreground'>Engagement Score:</div>
                <div className='text-base font-bold font-mono text-foreground'>{currentProfile.engagement} / 100</div>
              </div>
            </div>

            {/* Prescriptive Playbook Box */}
            <div className='rounded-lg border border-amber-300/80 bg-amber-50/70 dark:border-amber-900/50 dark:bg-amber-950/30 p-3.5'>
              <div className='flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-amber-800 dark:text-amber-400'>
                {getPlaybookIcon(currentProfile.playbook)}
                Recommended Retention Playbook
              </div>
              <div className='mt-1 text-sm font-semibold text-amber-950 dark:text-amber-200'>
                {currentProfile.playbook}
              </div>
              <div className='mt-1.5 text-[11px] text-amber-800/80 dark:text-amber-300/80 flex items-center gap-1.5'>
                <CheckCircle2 className='size-3 text-amber-600 shrink-0' />
                Trigger automated CRM task in HubSpot / Salesforce for Account Executive.
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Local SHAP Feature Drivers Bar Chart */}
      <Card className='border-border/60 shadow-sm'>
        <CardHeader>
          <div className='flex items-center justify-between'>
            <div>
              <CardTitle className='text-sm font-semibold flex items-center gap-2'>
                <Brain className='size-4 text-indigo-600' />
                Local SHAP Feature Attribution
              </CardTitle>
              <CardDescription className='text-xs'>
                Decomposition of exact factors pulling this account towards churn (+) or retention (-)
              </CardDescription>
            </div>
            <div className='flex items-center gap-3 text-xs'>
              <span className='flex items-center gap-1 text-rose-600 dark:text-rose-400 font-medium'>
                <span className='size-2 rounded-full bg-rose-500' />
                Pushing Toward Churn (+)
              </span>
              <span className='flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-medium'>
                <span className='size-2 rounded-full bg-emerald-500' />
                Supporting Retention (-)
              </span>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className='h-[340px] w-full'>
            <ResponsiveContainer width='100%' height='100%'>
              <BarChart data={currentProfile.drivers} layout='vertical' margin={{ top: 10, right: 30, left: 140, bottom: 10 }}>
                <CartesianGrid strokeDasharray='3 3' className='stroke-muted/40' horizontal={false} />
                <XAxis type='number' className='text-xs text-muted-foreground' stroke='currentColor' tickLine={false} />
                <YAxis
                  dataKey='feature'
                  type='category'
                  className='text-xs font-medium text-foreground'
                  stroke='currentColor'
                  tickLine={false}
                  width={130}
                  interval={0}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const item = payload[0].payload
                      return (
                        <div className='rounded-lg border border-border/60 bg-popover/95 p-2.5 shadow-md text-xs'>
                          <p className='font-semibold text-foreground mb-1'>{item.feature}</p>
                          <p className='text-muted-foreground'>Actual Feature Value: <b className='text-foreground'>{item.value}</b></p>
                          <p className={`font-mono font-bold ${item.shap > 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                            SHAP Contribution: {item.shap > 0 ? `+${item.shap}` : item.shap}
                          </p>
                        </div>
                      )
                    }
                    return null
                  }}
                />
                <Bar isAnimationActive={false} dataKey='shap' radius={[4, 4, 4, 4]} barSize={22}>
                  {currentProfile.drivers.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.shap > 0 ? '#EF4444' : '#10B981'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
