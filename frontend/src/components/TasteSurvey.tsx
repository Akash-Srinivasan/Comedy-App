'use client'
import { useState, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { createClient } from '@/lib/supabase'
import type { Database } from '@/lib/supabase'

type MediaItem = Database['public']['Tables']['media_items']['Row']

interface SurveyData {
  favoriteShows: string[]
  favoriteComedians: string[]
  humorStyles: string[]
  demographics: {
    ageRange: string
    city: string
    state: string
  }
}

interface TasteSurveyProps {
  onComplete: (data: any) => void
  userId: string
}

export function TasteSurvey({ onComplete, userId }: TasteSurveyProps) {
  const [currentStep, setCurrentStep] = useState(1)
  const [mediaItems, setMediaItems] = useState<MediaItem[]>([])
  const [selectedMedia, setSelectedMedia] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const supabase = createClient()

  const { register, handleSubmit, watch, setValue, formState: { errors } } = useForm<SurveyData>()

  useEffect(() => {
    loadMediaItems()
  }, [])

  const loadMediaItems = async () => {
    try {
      const { data, error } = await supabase
        .from('media_items')
        .select('*')
        .order('title')
      
      if (error) throw error
      if (data) setMediaItems(data)
    } catch (err) {
      console.error('Error loading media items:', err)
      setError('Failed to load media items')
    }
  }

  const calculateHumorDimensions = async (selectedMediaIds: string[]) => {
    if (selectedMediaIds.length === 0) return {}

    try {
      const { data: media, error } = await supabase
        .from('media_items')
        .select('humor_attributes')
        .in('id', selectedMediaIds)

      if (error) throw error
      if (!media || media.length === 0) return {}

      // Average the humor attributes across selected items
      const dimensions: Record<string, number> = {}
      const attributeKeys = new Set<string>()

      // Collect all attribute keys
      media.forEach(item => {
        if (item.humor_attributes && typeof item.humor_attributes === 'object') {
          Object.keys(item.humor_attributes).forEach(key => attributeKeys.add(key))
        }
      })

      // Calculate averages
      attributeKeys.forEach(key => {
        const values = media
          .map(item => {
            const attrs = item.humor_attributes as Record<string, any>
            return attrs?.[key]
          })
          .filter(val => typeof val === 'number')
        
        if (values.length > 0) {
          dimensions[key] = values.reduce((sum, val) => sum + val, 0) / values.length
        }
      })

      return dimensions
    } catch (err) {
      console.error('Error calculating humor dimensions:', err)
      return {}
    }
  }

  const onSubmit = async (data: SurveyData) => {
    setLoading(true)
    setError(null)

    try {
      // Calculate humor dimensions from selected media
      const humorDimensions = await calculateHumorDimensions(selectedMedia)
      
      // Save user profile
      const { error: profileError } = await supabase
        .from('user_profiles')
        .upsert({
          id: userId,
          age_range: data.demographics.ageRange,
          location_city: data.demographics.city,
          location_state: data.demographics.state,
          humor_dimensions: humorDimensions,
          preferences: {
            favorite_shows: selectedMedia,
            humor_styles: data.humorStyles || []
          },
          confidence_score: 0.3, // Initial confidence
          updated_at: new Date().toISOString()
        })

      if (profileError) throw profileError

      // Save media preferences
      if (selectedMedia.length > 0) {
        const mediaPreferences = selectedMedia.map(mediaId => ({
          user_id: userId,
          media_id: mediaId,
          preference_score: 4 // Assuming high preference for selected items
        }))

        const { error: preferencesError } = await supabase
          .from('user_media_preferences')
          .upsert(mediaPreferences)

        if (preferencesError) console.warn('Error saving media preferences:', preferencesError)
      }

      onComplete(humorDimensions)
    } catch (err) {
      console.error('Error saving survey data:', err)
      setError('Failed to save your preferences. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const toggleMediaSelection = (mediaId: string) => {
    setSelectedMedia(prev => 
      prev.includes(mediaId) 
        ? prev.filter(id => id !== mediaId)
        : prev.length < 8 ? [...prev, mediaId] : prev // Limit to 8 selections
    )
  }

  if (currentStep === 1) {
    return (
      <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow">
        <div className="mb-6">
          <h2 className="text-3xl font-bold mb-2">Tell us about your taste</h2>
          <p className="text-gray-600">Select 3-8 shows or movies you love. This helps us understand your humor preferences.</p>
        </div>
        
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-700">{error}</p>
          </div>
        )}
        
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-4">
            Which TV shows or movies do you love? ({selectedMedia.length}/8 selected)
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {mediaItems.map((item) => (
              <div
                key={item.id}
                onClick={() => toggleMediaSelection(item.id)}
                className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  selectedMedia.includes(item.id)
                    ? 'border-blue-500 bg-blue-50 transform scale-105'
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                } ${selectedMedia.length >= 8 && !selectedMedia.includes(item.id) ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <div className="text-sm font-medium mb-1">{item.title}</div>
                <div className="text-xs text-gray-500">
                  {item.year} • {item.type?.replace('_', ' ')}
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  {item.genres?.slice(0, 2).join(', ')}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-between items-center">
          <div className="text-sm text-gray-500">
            Select at least 3 items to continue
          </div>
          <button
            onClick={() => setCurrentStep(2)}
            disabled={selectedMedia.length < 3}
            className="bg-blue-600 text-white py-3 px-6 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors font-medium"
          >
            Next Step →
          </button>
        </div>
      </div>
    )
  }

  if (currentStep === 2) {
    return (
      <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow">
        <div className="mb-6">
          <h2 className="text-3xl font-bold mb-2">Comedy Preferences</h2>
          <p className="text-gray-600">Help us fine-tune your recommendations with a few more details.</p>
        </div>
        
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold mb-4">What comedy styles do you enjoy? (Select all that apply)</h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'observational', label: 'Observational' },
                { value: 'storytelling', label: 'Storytelling' },
                { value: 'political', label: 'Political' },
                { value: 'absurd', label: 'Absurd/Surreal' },
                { value: 'dark', label: 'Dark Humor' },
                { value: 'clean', label: 'Clean/Family-Friendly' },
                { value: 'relationships', label: 'Relationship Comedy' },
                { value: 'workplace', label: 'Workplace Comedy' }
              ].map((style) => (
                <label key={style.value} className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 p-2 rounded">
                  <input
                    type="checkbox"
                    {...register('humorStyles')}
                    value={style.value}
                    className="rounded border-gray-300"
                  />
                  <span>{style.label}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Age Range *</label>
              <select 
                {...register('demographics.ageRange', { required: 'Age range is required' })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Select age</option>
                <option value="18-24">18-24</option>
                <option value="25-34">25-34</option>
                <option value="35-44">35-44</option>
                <option value="45-54">45-54</option>
                <option value="55-64">55-64</option>
                <option value="65+">65+</option>
              </select>
              {errors.demographics?.ageRange && (
                <p className="text-red-500 text-sm mt-1">{errors.demographics.ageRange.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">City *</label>
              <input
                {...register('demographics.city', { required: 'City is required' })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Washington"
              />
              {errors.demographics?.city && (
                <p className="text-red-500 text-sm mt-1">{errors.demographics.city.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">State *</label>
              <input
                {...register('demographics.state', { 
                  required: 'State is required',
                  maxLength: { value: 2, message: 'Use 2-letter state code' },
                  minLength: { value: 2, message: 'Use 2-letter state code' }
                })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="DC"
                maxLength={2}
                style={{ textTransform: 'uppercase' }}
              />
              {errors.demographics?.state && (
                <p className="text-red-500 text-sm mt-1">{errors.demographics.state.message}</p>
              )}
            </div>
          </div>

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-700">{error}</p>
            </div>
          )}

          <div className="flex space-x-4">
            <button
              type="button"
              onClick={() => setCurrentStep(1)}
              disabled={loading}
              className="flex-1 bg-gray-200 text-gray-700 py-3 px-4 rounded-lg hover:bg-gray-300 transition-colors font-medium disabled:opacity-50"
            >
              ← Back
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Saving...' : 'Complete Survey'}
            </button>
          </div>
        </form>
      </div>
    )
  }

  return null
}